"""E2E test: real YouTube download via yt-dlp + ffmpeg audio extraction.

These tests hit the network and require yt-dlp + ffmpeg on the machine.
Skipped by default; run with ``pytest --run-slow``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from yt_dbl.config import Settings
from yt_dbl.pipeline.download import DownloadStep
from yt_dbl.schemas import PipelineState, StepName, StepStatus
from yt_dbl.utils.audio import get_audio_duration

if TYPE_CHECKING:
    from pathlib import Path

from tests.conftest import SHORT_VIDEO_ID, SHORT_VIDEO_URL

pytestmark = pytest.mark.slow


@pytest.fixture
def _require_tools(ytdlp_available: bool, ffmpeg_available: bool) -> None:
    if not ytdlp_available:
        pytest.skip("yt-dlp not installed")
    if not ffmpeg_available:
        pytest.skip("ffmpeg not installed")


@pytest.mark.usefixtures("_require_tools")
class TestE2EDownload:
    """Full download → extract cycle on a real short YouTube video."""

    def _make(
        self, work_dir: Path, video_id: str = SHORT_VIDEO_ID, url: str = SHORT_VIDEO_URL
    ) -> tuple[DownloadStep, Settings, PipelineState]:
        cfg = Settings(work_dir=work_dir)
        step_dir = cfg.step_dir(video_id, "01_download")
        step = DownloadStep(settings=cfg, work_dir=step_dir)
        state = PipelineState(video_id=video_id, url=url)
        return step, cfg, state

    def test_download_and_extract(self, e2e_work_dir: Path) -> None:
        """Download "Me at the zoo" and verify artefacts."""
        step, _cfg, state = self._make(e2e_work_dir)
        state = step.run(state)

        video_path = step.step_dir / "video.mp4"
        audio_path = step.step_dir / "audio.wav"

        # Files exist and are non-trivial
        assert video_path.exists()
        assert audio_path.exists()
        assert video_path.stat().st_size > 10_000, "Video too small"
        assert audio_path.stat().st_size > 10_000, "Audio too small"

        # Metadata populated
        assert state.meta is not None
        assert state.meta.title != ""
        assert state.meta.duration > 10  # ~19s video

        # Audio duration roughly matches video
        audio_dur = get_audio_duration(audio_path)
        assert abs(audio_dur - state.meta.duration) < 2.0

        # Step result outputs recorded
        result = state.get_step(StepName.DOWNLOAD)
        assert result.outputs["video"] == "video.mp4"
        assert result.outputs["audio"] == "audio.wav"

    def test_idempotent_rerun_skips(self, e2e_work_dir: Path) -> None:
        """Running the step twice should skip download on second run."""
        step, _cfg, state = self._make(e2e_work_dir)

        # First run — real download
        state = step.run(state)
        video_path = step.step_dir / "video.mp4"
        mtime_first = video_path.stat().st_mtime

        # Reset state step to pending so we can re-run
        state.get_step(StepName.DOWNLOAD).status = StepStatus.PENDING
        state.get_step(StepName.DOWNLOAD).outputs = {}

        # Second run — should skip (files exist)
        state2 = PipelineState(video_id=state.video_id, url=state.url)
        state2.meta = None  # force metadata re-fetch (it's fast)
        state2 = step.run(state2)

        # File should NOT have been re-downloaded (mtime unchanged)
        mtime_second = video_path.stat().st_mtime
        assert mtime_first == mtime_second

    def test_audio_is_mono_48khz(self, e2e_work_dir: Path) -> None:
        """Verify extracted audio properties: mono, 48kHz WAV."""
        import subprocess

        step, _cfg, state = self._make(e2e_work_dir)
        state = step.run(state)

        audio_path = step.step_dir / "audio.wav"

        # Use ffprobe to check audio properties
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=channels,sample_rate,codec_name",
                "-of",
                "csv=p=0",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        parts = result.stdout.strip().split(",")
        codec = parts[0]
        sample_rate = parts[1]
        channels = parts[2]

        assert codec == "pcm_s16le", f"Expected WAV PCM, got {codec}"
        assert sample_rate == "48000", f"Expected 48000 Hz, got {sample_rate}"
        assert channels == "1", f"Expected mono (1 ch), got {channels}"

    def test_invalid_url_raises(self, e2e_work_dir: Path) -> None:
        """DownloadError should be raised for a bogus URL."""
        from yt_dbl.pipeline.download import DownloadError

        step, _cfg, state = self._make(
            e2e_work_dir,
            video_id="INVALID",
            url="https://www.youtube.com/watch?v=INVALID_VIDEO_ID_XYZ_123",
        )
        with pytest.raises(DownloadError):
            step.run(state)
