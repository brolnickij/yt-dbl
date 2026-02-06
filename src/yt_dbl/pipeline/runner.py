"""Pipeline runner — orchestrates steps, checkpoints, and resume logic."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from yt_dbl.models.manager import ModelManager
from yt_dbl.schemas import (
    STEP_DIRS,
    STEP_ORDER,
    PipelineState,
    StepName,
    StepStatus,
)
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
from .download import DownloadStep
from .separate import SeparateStep
from .synthesize import SynthesizeStep
from .transcribe import TranscribeStep
from .translate import TranslateStep

if TYPE_CHECKING:
    from pathlib import Path

    from yt_dbl.config import Settings

    from .base import PipelineStep

# ── State persistence ───────────────────────────────────────────────────────

STATE_FILE = "state.json"


def _state_path(settings: Settings, video_id: str) -> Path:
    return settings.job_dir(video_id) / STATE_FILE


def save_state(state: PipelineState, settings: Settings) -> Path:
    """Persist pipeline state to disk."""
    path = _state_path(settings, state.video_id)
    path.write_text(state.model_dump_json(), encoding="utf-8")
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

        started = not from_step

        for step_name in STEP_ORDER:
            # Skip until we reach from_step
            if not started:
                if step_name == from_step:
                    started = True
                else:
                    result = state.get_step(step_name)
                    if result.status == StepStatus.COMPLETED:
                        log_step_skip(step_name)
                        continue

            # Skip already completed steps (when resuming without from_step)
            if from_step is None:
                result = state.get_step(step_name)
                if result.status == StepStatus.COMPLETED:
                    log_step_skip(step_name)
                    continue

            # Run the step
            state = self._run_step(step_name, state)

            # Stop on failure
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

        # Free all models at the end of pipeline
        self.model_manager.unload_all()
        log_memory_status()

        return state

    def _run_step(self, step_name: StepName, state: PipelineState) -> PipelineState:
        """Run a single pipeline step with timing and checkpointing."""
        step_dir = self.settings.step_dir(state.video_id, STEP_DIRS[step_name])
        step_cls = STEP_CLASSES[step_name]
        step = step_cls(
            settings=self.settings,
            work_dir=step_dir,
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

        except Exception as exc:
            result.status = StepStatus.FAILED
            result.finished_at = datetime.now(UTC).isoformat()
            result.error = str(exc)
            log_step_fail(step_name, str(exc))

        save_state(state, self.settings)
        return state
