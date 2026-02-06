"""E2E test: real transcription with VibeVoice-ASR + ForcedAligner.

Downloads a short YouTube video, separates vocals, then runs real
ASR + diarization + word-level alignment on Apple Silicon MLX Metal.

Skipped by default; run with ``pytest --run-slow``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.conftest import SHORT_VIDEO_ID, SHORT_VIDEO_URL
from yt_dbl.config import Settings
from yt_dbl.pipeline.download import DownloadStep
from yt_dbl.pipeline.separate import SeparateStep
from yt_dbl.pipeline.transcribe import TranscribeStep
from yt_dbl.schemas import STEP_DIRS, PipelineState, StepName, StepStatus

if TYPE_CHECKING:
    from pathlib import Path

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


def _download_and_separate(
    work_dir: Path,
    video_id: str = SHORT_VIDEO_ID,
    url: str = SHORT_VIDEO_URL,
) -> tuple[Settings, PipelineState]:
    """Run download + separate so transcription has vocals input."""
    cfg = Settings(work_dir=work_dir)
    state = PipelineState(video_id=video_id, url=url)

    dl_dir = cfg.step_dir(video_id, STEP_DIRS[StepName.DOWNLOAD])
    dl_step = DownloadStep(settings=cfg, work_dir=dl_dir)
    state = dl_step.run(state)

    sep_dir = cfg.step_dir(video_id, STEP_DIRS[StepName.SEPARATE])
    sep_step = SeparateStep(settings=cfg, work_dir=sep_dir)
    state = sep_step.run(state)

    return cfg, state


@pytest.mark.usefixtures("_require_tools")
class TestE2ETranscribe:
    """Real transcription on a short YouTube video."""

    @pytest.mark.timeout(600)
    def test_transcribe_produces_segments(self, e2e_work_dir: Path) -> None:
        """Transcribe 'Me at the zoo' and verify segments + speakers."""
        cfg, state = _download_and_separate(e2e_work_dir)

        trans_dir = cfg.step_dir(SHORT_VIDEO_ID, STEP_DIRS[StepName.TRANSCRIBE])
        step = TranscribeStep(settings=cfg, work_dir=trans_dir)
        state = step.run(state)

        # Step marked complete with outputs
        result = state.get_step(StepName.TRANSCRIBE)
        assert result.outputs.get("segments") == "segments.json"

        # Segments were produced
        assert len(state.segments) > 0, "No segments produced"

        for seg in state.segments:
            # Each segment has basic fields
            assert seg.text.strip(), f"Segment {seg.id} has empty text"
            assert seg.end > seg.start, f"Segment {seg.id} has invalid time range"
            assert seg.speaker.startswith("SPEAKER_"), f"Bad speaker id: {seg.speaker}"

            # Word-level alignment produced words
            assert len(seg.words) > 0, f"Segment {seg.id} has no words"
            for word in seg.words:
                assert word.text.strip(), "Empty word text"

        # Speakers were extracted
        assert len(state.speakers) > 0, "No speakers detected"
        for spk in state.speakers:
            assert spk.total_duration > 0.0, f"Speaker {spk.id} has zero duration"

        # Segments JSON file was persisted
        segments_json = trans_dir / "segments.json"
        assert segments_json.exists(), "segments.json was not saved"
        assert segments_json.stat().st_size > 100, "segments.json is suspiciously small"

    @pytest.mark.timeout(600)
    def test_idempotent_rerun_uses_cache(self, e2e_work_dir: Path) -> None:
        """Running transcription twice reuses cached segments.json."""
        cfg, state = _download_and_separate(e2e_work_dir)

        trans_dir = cfg.step_dir(SHORT_VIDEO_ID, STEP_DIRS[StepName.TRANSCRIBE])
        step = TranscribeStep(settings=cfg, work_dir=trans_dir)

        # First run — real transcription
        state = step.run(state)
        segments_json = trans_dir / "segments.json"
        mtime_first = segments_json.stat().st_mtime
        segments_count = len(state.segments)

        # Reset step status
        state.get_step(StepName.TRANSCRIBE).status = StepStatus.PENDING
        state.get_step(StepName.TRANSCRIBE).outputs = {}

        # Second run — should load from cache (no model loading)
        state = step.run(state)
        mtime_second = segments_json.stat().st_mtime

        assert mtime_first == mtime_second, "segments.json was re-created on rerun"
        assert len(state.segments) == segments_count, "Segment count changed on rerun"
