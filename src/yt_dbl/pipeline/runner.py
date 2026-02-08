"""Pipeline runner — orchestrates steps, checkpoints, and resume logic."""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from yt_dbl.models.manager import ModelManager
from yt_dbl.schemas import (
    STEP_DIRS,
    STEP_ORDER,
    PipelineState,
    StepName,
    StepStatus,
)
from yt_dbl.utils.audio import set_ffmpeg_path
from yt_dbl.utils.logging import (
    console,
    log_info,
    log_memory_status,
    log_step_done,
    log_step_fail,
    log_step_skip,
    log_step_start,
    timer,
)

from .assemble import AssembleStep
from .base import StepValidationError
from .download import DownloadStep
from .separate import SeparateStep
from .synthesize import SynthesizeStep
from .transcribe import TranscribeStep
from .translate import TranslateStep

if TYPE_CHECKING:
    from yt_dbl.config import Settings

    from .base import PipelineStep

# ── State persistence ───────────────────────────────────────────────────────

STATE_FILE = "state.json"


def _state_path(settings: Settings, video_id: str) -> Path:
    return settings.job_dir(video_id) / STATE_FILE


def save_state(state: PipelineState, settings: Settings) -> Path:
    """Persist pipeline state to disk atomically.

    Writes to a temporary file in the same directory and then replaces
    the target via ``os.replace`` — which is atomic on POSIX.  This
    prevents partial/corrupt ``state.json`` if the process is killed
    mid-write.
    """
    path = _state_path(settings, state.video_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_str = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    tmp = Path(tmp_str)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(state.model_dump_json())
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise
    return path


def load_state(settings: Settings, video_id: str) -> PipelineState | None:
    """Load pipeline state from disk, or None if not found."""
    path = _state_path(settings, video_id)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return PipelineState.model_validate(data)


# ── Step registry ───────────────────────────────────────────────────────────

STEP_CLASSES: dict[StepName, type[PipelineStep]] = {
    StepName.DOWNLOAD: DownloadStep,
    StepName.SEPARATE: SeparateStep,
    StepName.TRANSCRIBE: TranscribeStep,
    StepName.TRANSLATE: TranslateStep,
    StepName.SYNTHESIZE: SynthesizeStep,
    StepName.ASSEMBLE: AssembleStep,
}


# ── Runner ──────────────────────────────────────────────────────────────────


class PipelineRunner:
    """Orchestrates the full dubbing pipeline with checkpointing."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model_manager = ModelManager(max_loaded=settings.max_loaded_models)
        set_ffmpeg_path(settings.ffmpeg_path)

    def run(
        self,
        state: PipelineState,
        from_step: StepName | None = None,
    ) -> PipelineState:
        """Run the pipeline from the given step (or from where it left off)."""
        console.rule(f"[bold]yt-dbl — {state.url}[/bold]")
        log_info(f"Video ID: {state.video_id}")
        log_info(f"Target language: {state.target_language}")
        log_info(f"Max models in memory: {self.model_manager.max_loaded}")
        log_info(f"Work dir: {self.settings.job_dir(state.video_id)}")
        console.print()

        # Early validation: API key is needed unless translate step is already done
        translate_result = state.get_step(StepName.TRANSLATE)
        if translate_result.status != StepStatus.COMPLETED:
            will_run = from_step is None or STEP_ORDER.index(from_step) <= STEP_ORDER.index(
                StepName.TRANSLATE
            )
            if will_run and not self.settings.anthropic_api_key:
                raise StepValidationError(
                    "Anthropic API key required for translation — set YT_DBL_ANTHROPIC_API_KEY"
                )

        steps_to_run = self._resolve_steps_to_run(state, from_step)

        try:
            for step_name in steps_to_run:
                state = self._run_step(step_name, state)

                if state.get_step(step_name).status == StepStatus.FAILED:
                    log_info("Pipeline stopped due to failure. Use 'resume' to retry.")
                    break

            console.print()
            if state.next_step is None:
                console.rule("[bold green]Done![/bold green]")
                outputs = state.get_step(StepName.ASSEMBLE).outputs
                if "result" in outputs:
                    result_path = self.settings.job_dir(state.video_id) / outputs["result"]
                    console.print(f"  Result: [bold]{result_path}[/bold]")
            else:
                console.rule("[bold yellow]Incomplete[/bold yellow]")
        finally:
            # Free all models even on KeyboardInterrupt / unexpected errors
            self.model_manager.unload_all()
            log_memory_status()

        return state

    @staticmethod
    def _resolve_steps_to_run(
        state: PipelineState,
        from_step: StepName | None,
    ) -> list[StepName]:
        """Determine which pipeline steps need to run, logging skipped ones."""
        steps: list[StepName] = []
        started = not from_step

        for step_name in STEP_ORDER:
            if not started:
                if step_name == from_step:
                    started = True
                else:
                    if state.get_step(step_name).status == StepStatus.COMPLETED:
                        log_step_skip(step_name)
                    continue

            if from_step is None and state.get_step(step_name).status == StepStatus.COMPLETED:
                log_step_skip(step_name)
                continue

            steps.append(step_name)

        return steps

    def _run_step(self, step_name: StepName, state: PipelineState) -> PipelineState:
        """Run a single pipeline step with timing and checkpointing."""
        # Free GPU memory before assembly — no ML models are needed and the
        # speech-track builder benefits from the headroom (especially on
        # 16 GB machines where the TTS model would otherwise remain loaded).
        if step_name == StepName.ASSEMBLE:
            self.model_manager.unload_all()

        step_dir = self.settings.step_dir(state.video_id, STEP_DIRS[step_name])
        step_cls = STEP_CLASSES[step_name]
        step = step_cls(
            settings=self.settings,
            step_dir=step_dir,
            model_manager=self.model_manager,
        )

        result = state.get_step(step_name)
        result.status = StepStatus.RUNNING
        result.started_at = datetime.now(UTC).isoformat()
        save_state(state, self.settings)

        log_step_start(step_name, step.description)

        try:
            step.validate_inputs(state)

            with timer() as t:
                state = step.run(state)

            elapsed = t["elapsed"]
            result.status = StepStatus.COMPLETED
            result.finished_at = datetime.now(UTC).isoformat()
            result.duration_sec = elapsed
            result.error = ""

            log_step_done(step_name, elapsed)

        except StepValidationError as exc:
            # Validation errors are fatal: persist state and re-raise so the
            # caller sees the same exception type as the early validation in
            # ``run()`` — instead of silently swallowing it as a generic FAILED.
            result.status = StepStatus.FAILED
            result.finished_at = datetime.now(UTC).isoformat()
            result.error = str(exc)
            log_step_fail(step_name, str(exc))
            save_state(state, self.settings)
            raise

        except Exception as exc:
            result.status = StepStatus.FAILED
            result.finished_at = datetime.now(UTC).isoformat()
            result.error = str(exc)
            log_step_fail(step_name, str(exc))

        save_state(state, self.settings)
        return state
