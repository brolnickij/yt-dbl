"""E2E test: full pipeline run — real download + separation + transcription, stub steps 4-6.

Verifies that PipelineRunner drives all steps end-to-end:
  download (real) → separate (real) → transcribe (real) →
  translate (stub) → synthesize (stub) → assemble (stub)

Skipped by default; run with ``pytest --run-slow``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from yt_dbl.config import Settings
from yt_dbl.pipeline.runner import PipelineRunner, load_state, save_state
from yt_dbl.schemas import PipelineState, StepName, StepStatus

if TYPE_CHECKING:
    from pathlib import Path

from tests.conftest import SHORT_VIDEO_ID, SHORT_VIDEO_URL

pytestmark = pytest.mark.slow


@pytest.fixture
def _require_tools(
    ytdlp_available: bool,
    ffmpeg_available: bool,
    audio_separator_available: bool,
) -> None:
    if not ytdlp_available:
        pytest.skip("yt-dlp not installed")
    if not ffmpeg_available:
        pytest.skip("ffmpeg not installed")
    if not audio_separator_available:
        pytest.skip("audio-separator not installed")


@pytest.mark.usefixtures("_require_tools")
class TestE2EPipeline:
    """Full pipeline traversal with real download + separation, stub steps 3-6."""

    def test_full_pipeline_run(self, e2e_work_dir: Path) -> None:
        """Run entire pipeline on a real short video."""
        cfg = Settings(work_dir=e2e_work_dir)
        state = PipelineState(
            video_id=SHORT_VIDEO_ID,
            url=SHORT_VIDEO_URL,
            target_language="ru",
        )

        runner = PipelineRunner(cfg)
        state = runner.run(state)

        # All steps completed
        for step_name in StepName:
            assert state.get_step(step_name).status == StepStatus.COMPLETED, (
                f"{step_name} not completed"
            )

        # No next step — pipeline finished
        assert state.next_step is None

        # Download artefacts exist
        dl_dir = cfg.step_dir(SHORT_VIDEO_ID, "01_download")
        assert (dl_dir / "video.mp4").exists()
        assert (dl_dir / "audio.wav").exists()

        # State checkpoint was persisted
        loaded = load_state(cfg, SHORT_VIDEO_ID)
        assert loaded is not None
        assert loaded.meta is not None
        assert loaded.meta.title != ""

    def test_resume_from_translate(self, e2e_work_dir: Path) -> None:
        """Run pipeline, then resume from translate step."""
        cfg = Settings(work_dir=e2e_work_dir)
        state = PipelineState(
            video_id=SHORT_VIDEO_ID,
            url=SHORT_VIDEO_URL,
            target_language="ru",
        )

        runner = PipelineRunner(cfg)
        state = runner.run(state)

        # Pretend translate/synthesize/assemble need to be re-run
        for name in [StepName.TRANSLATE, StepName.SYNTHESIZE, StepName.ASSEMBLE]:
            state.get_step(name).status = StepStatus.PENDING
        save_state(state, cfg)

        # Resume — should only re-run translate+synth+assemble, NOT download again
        loaded = load_state(cfg, SHORT_VIDEO_ID)
        assert loaded is not None
        assert loaded.next_step == StepName.TRANSLATE

        state = runner.run(loaded)
        assert state.next_step is None
        assert state.get_step(StepName.TRANSLATE).status == StepStatus.COMPLETED
