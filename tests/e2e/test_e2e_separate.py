"""E2E test: real vocal separation with BS-RoFormer via audio-separator.

Downloads a short YouTube video, then runs real separation.
Skipped by default; run with ``pytest --run-slow``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from yt_dbl.config import Settings
from yt_dbl.pipeline.download import DownloadStep
from yt_dbl.pipeline.separate import SeparateStep
from yt_dbl.schemas import PipelineState, StepName, StepStatus
from yt_dbl.utils.audio import get_audio_duration

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


def _download_first(
    work_dir: Path,
    video_id: str = SHORT_VIDEO_ID,
    url: str = SHORT_VIDEO_URL,
) -> tuple[Settings, PipelineState]:
    """Run download step so separation has input audio."""
    cfg = Settings(work_dir=work_dir)
    state = PipelineState(video_id=video_id, url=url)

    dl_dir = cfg.step_dir(video_id, "01_download")
    dl_step = DownloadStep(settings=cfg, work_dir=dl_dir)
    state = dl_step.run(state)
    return cfg, state


@pytest.mark.usefixtures("_require_tools")
class TestE2ESeparate:
    """Full separation cycle on a real downloaded audio file."""

    def test_separate_produces_vocals_and_background(self, e2e_work_dir: Path) -> None:
        """Separate "Me at the zoo" and verify both stems are created."""
        cfg, state = _download_first(e2e_work_dir)

        sep_dir = cfg.step_dir(SHORT_VIDEO_ID, "02_separate")
        step = SeparateStep(settings=cfg, work_dir=sep_dir)
        state = step.run(state)

        vocals_path = sep_dir / "vocals.wav"
        background_path = sep_dir / "background.wav"

        # Both files exist and are non-trivial
        assert vocals_path.exists(), "vocals.wav was not created"
        assert background_path.exists(), "background.wav was not created"
        assert vocals_path.stat().st_size > 10_000, "Vocals file is too small"
        assert background_path.stat().st_size > 10_000, "Background file is too small"

        # Step outputs are recorded correctly
        result = state.get_step(StepName.SEPARATE)
        assert result.outputs["vocals"] == "vocals.wav"
        assert result.outputs["background"] == "background.wav"

    def test_stems_are_valid_audio(self, e2e_work_dir: Path) -> None:
        """Verify that separated stems are valid WAV files with plausible durations."""
        cfg, state = _download_first(e2e_work_dir)

        sep_dir = cfg.step_dir(SHORT_VIDEO_ID, "02_separate")
        step = SeparateStep(settings=cfg, work_dir=sep_dir)
        state = step.run(state)

        assert state.meta is not None
        original_duration = state.meta.duration

        vocals_dur = get_audio_duration(sep_dir / "vocals.wav")
        bg_dur = get_audio_duration(sep_dir / "background.wav")

        # Stems should be roughly the same duration as the original (±2 s)
        assert abs(vocals_dur - original_duration) < 2.0, (
            f"Vocals duration {vocals_dur:.1f}s too far from original {original_duration:.1f}s"
        )
        assert abs(bg_dur - original_duration) < 2.0, (
            f"Background duration {bg_dur:.1f}s too far from original {original_duration:.1f}s"
        )

    def test_idempotent_rerun_skips(self, e2e_work_dir: Path) -> None:
        """Running separation twice reuses cached outputs."""
        cfg, state = _download_first(e2e_work_dir)

        sep_dir = cfg.step_dir(SHORT_VIDEO_ID, "02_separate")
        step = SeparateStep(settings=cfg, work_dir=sep_dir)

        # First run — real separation
        state = step.run(state)
        vocals_path = sep_dir / "vocals.wav"
        mtime_first = vocals_path.stat().st_mtime

        # Reset step status
        state.get_step(StepName.SEPARATE).status = StepStatus.PENDING
        state.get_step(StepName.SEPARATE).outputs = {}

        # Second run — should skip (outputs exist)
        state = step.run(state)
        mtime_second = vocals_path.stat().st_mtime

        assert mtime_first == mtime_second, "Vocals file was re-created on idempotent rerun"

    def test_no_intermediate_files_left(self, e2e_work_dir: Path) -> None:
        """After separation, only vocals.wav and background.wav should remain."""
        cfg, state = _download_first(e2e_work_dir)

        sep_dir = cfg.step_dir(SHORT_VIDEO_ID, "02_separate")
        step = SeparateStep(settings=cfg, work_dir=sep_dir)
        step.run(state)

        wav_files = sorted(f.name for f in sep_dir.glob("*.wav"))
        assert wav_files == ["background.wav", "vocals.wav"], (
            f"Unexpected files in separation dir: {wav_files}"
        )
