"""Tests for yt_dbl.pipeline.assemble — final video assembly."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from yt_dbl.config import Settings
from yt_dbl.pipeline.assemble import (
    SPEECH_TRACK_FILE,
    AssembleStep,
    _assemble_video,
    _build_speech_track,
)
from yt_dbl.schemas import (
    PipelineState,
    Segment,
    Speaker,
    StepName,
    StepStatus,
    VideoMeta,
    Word,
)

if TYPE_CHECKING:
    from pathlib import Path


# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_segments() -> list[Segment]:
    return [
        Segment(
            id=0,
            text="Hello, welcome to this video.",
            start=0.5,
            end=3.5,
            speaker="SPEAKER_00",
            language="en",
            translated_text="Привет, добро пожаловать.",
            synth_path="segment_0000.wav",
            words=[Word(text="Hello,", start=0.5, end=1.0)],
        ),
        Segment(
            id=1,
            text="Today we talk about something.",
            start=4.0,
            end=7.0,
            speaker="SPEAKER_00",
            language="en",
            translated_text="Сегодня поговорим.",
            synth_path="segment_0001.wav",
        ),
        Segment(
            id=2,
            text="That sounds great!",
            start=8.0,
            end=10.0,
            speaker="SPEAKER_01",
            language="en",
            translated_text="Звучит отлично!",
            synth_path="segment_0002.wav",
        ),
    ]


def _make_speakers() -> list[Speaker]:
    return [
        Speaker(id="SPEAKER_00", total_duration=6.0),
        Speaker(id="SPEAKER_01", total_duration=2.0),
    ]


def _write_wav(path: Path, duration_sec: float = 1.0, sample_rate: int = 48000) -> None:
    """Write a tiny sine-wave WAV for testing."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), dtype=np.float32)
    data = 0.5 * np.sin(2 * np.pi * 440 * t)
    sf.write(str(path), data, sample_rate)


def _make_step(tmp_path: Path) -> tuple[AssembleStep, Settings, PipelineState]:
    """Create an AssembleStep with a fully prefilled PipelineState."""
    cfg = Settings(work_dir=tmp_path / "work")
    step_dir = cfg.step_dir("test123", "06_assemble")
    step = AssembleStep(settings=cfg, work_dir=step_dir)

    state = PipelineState(
        video_id="test123",
        url="https://example.com",
        meta=VideoMeta(
            video_id="test123",
            title="Test",
            channel="Test",
            duration=12.0,
            url="https://example.com",
        ),
    )
    state.segments = _make_segments()
    state.speakers = _make_speakers()

    # Prefill download step
    dl = state.get_step(StepName.DOWNLOAD)
    dl.status = StepStatus.COMPLETED
    dl.outputs = {"video": "video.mp4", "audio": "audio.wav"}
    dl_dir = cfg.step_dir("test123", "01_download")
    (dl_dir / "video.mp4").write_bytes(b"fake-video")

    # Prefill separate step
    sep = state.get_step(StepName.SEPARATE)
    sep.status = StepStatus.COMPLETED
    sep.outputs = {"vocals": "vocals.wav", "background": "background.wav"}
    sep_dir = cfg.step_dir("test123", "02_separate")
    _write_wav(sep_dir / "background.wav", duration_sec=12.0)

    # Prefill translate step
    trans = state.get_step(StepName.TRANSLATE)
    trans.status = StepStatus.COMPLETED
    trans.outputs = {"translations": "translations.json", "subtitles": "subtitles.srt"}
    trans_dir = cfg.step_dir("test123", "04_translate")
    (trans_dir / "subtitles.srt").write_text(
        "1\n00:00:00,500 --> 00:00:03,500\nПривет\n", encoding="utf-8"
    )

    # Prefill synthesize step — create actual WAV files
    synth = state.get_step(StepName.SYNTHESIZE)
    synth.status = StepStatus.COMPLETED
    synth.outputs = {f"seg_{seg.id}": seg.synth_path for seg in state.segments}
    synth_dir = cfg.step_dir("test123", "05_synthesize")
    for seg in state.segments:
        _write_wav(synth_dir / seg.synth_path, duration_sec=seg.duration)

    return step, cfg, state


# ── Validation tests ────────────────────────────────────────────────────────


class TestAssembleValidation:
    def test_validate_no_synth_outputs(self, tmp_path: Path) -> None:
        step, _, state = _make_step(tmp_path)
        state.get_step(StepName.SYNTHESIZE).outputs = {}
        with pytest.raises(ValueError, match="No synthesized segments"):
            step.validate_inputs(state)

    def test_validate_no_video(self, tmp_path: Path) -> None:
        step, _, state = _make_step(tmp_path)
        state.get_step(StepName.DOWNLOAD).outputs = {}
        with pytest.raises(ValueError, match="No video file"):
            step.validate_inputs(state)

    def test_validate_no_background(self, tmp_path: Path) -> None:
        step, _, state = _make_step(tmp_path)
        state.get_step(StepName.SEPARATE).outputs = {}
        with pytest.raises(ValueError, match="No background audio"):
            step.validate_inputs(state)

    def test_validate_ok(self, tmp_path: Path) -> None:
        step, _, state = _make_step(tmp_path)
        step.validate_inputs(state)  # should not raise


# ── Speech track tests ──────────────────────────────────────────────────────


class TestBuildSpeechTrack:
    def test_places_segments_at_timecodes(self, tmp_path: Path) -> None:
        """Segments must appear at their start positions, not at zero."""
        segments = _make_segments()
        synth_dir = tmp_path / "synth"
        synth_dir.mkdir()
        for seg in segments:
            _write_wav(synth_dir / seg.synth_path, duration_sec=seg.duration)

        track = _build_speech_track(segments, synth_dir, total_duration=12.0)

        # Before first segment (~0.5s) should be silence
        silence_region = track[: int(0.4 * 48000)]
        assert np.all(silence_region == 0.0)

        # At segment 0 start (0.5s) should have audio
        seg0_region = track[int(0.6 * 48000) : int(1.0 * 48000)]
        assert np.any(seg0_region != 0.0)

        # Between segments (3.5-4.0s) should be silence
        gap_region = track[int(3.6 * 48000) : int(3.9 * 48000)]
        assert np.all(gap_region == 0.0)

    def test_applies_crossfade(self, tmp_path: Path) -> None:
        """First and last samples of each segment should be faded."""
        segments = [
            Segment(
                id=0,
                text="Test",
                start=0.0,
                end=1.0,
                translated_text="Тест",
                synth_path="seg.wav",
            ),
        ]
        synth_dir = tmp_path / "synth"
        synth_dir.mkdir()
        # Create 1s of constant amplitude
        sr = 48000
        data = np.ones(sr, dtype=np.float32) * 0.8
        sf.write(str(synth_dir / "seg.wav"), data, sr)

        track = _build_speech_track(segments, synth_dir, total_duration=2.0)

        # First sample should be near zero (faded in)
        assert abs(track[0]) < 0.01
        # Middle should be full amplitude
        mid = sr // 2
        assert abs(track[mid] - 0.8) < 0.01
        # Last sample of segment should be near zero (faded out)
        assert abs(track[sr - 1]) < 0.01

    def test_skips_missing_segments(self, tmp_path: Path) -> None:
        """Segments with no synth_path or missing files are skipped."""
        segments = [
            Segment(id=0, text="A", start=0.0, end=1.0, synth_path=""),
            Segment(id=1, text="B", start=2.0, end=3.0, synth_path="missing.wav"),
        ]
        synth_dir = tmp_path / "synth"
        synth_dir.mkdir()

        track = _build_speech_track(segments, synth_dir, total_duration=5.0)
        assert np.all(track == 0.0)

    def test_clips_to_total_duration(self, tmp_path: Path) -> None:
        """Segments extending past total_duration are clipped."""
        segments = [
            Segment(
                id=0,
                text="Long",
                start=0.0,
                end=5.0,
                translated_text="Длинный",
                synth_path="long.wav",
            ),
        ]
        synth_dir = tmp_path / "synth"
        synth_dir.mkdir()
        _write_wav(synth_dir / "long.wav", duration_sec=5.0)

        # total_duration is only 2s — segment should be clipped
        track = _build_speech_track(segments, synth_dir, total_duration=2.0)
        expected_samples = int(2.0 * 48000) + 48000  # +1s safety
        assert len(track) == expected_samples

    def test_resamples_if_needed(self, tmp_path: Path) -> None:
        """Files at different sample rates are resampled to target."""
        segments = [
            Segment(
                id=0,
                text="X",
                start=0.0,
                end=0.5,
                translated_text="Х",
                synth_path="24k.wav",
            ),
        ]
        synth_dir = tmp_path / "synth"
        synth_dir.mkdir()
        # Write at 24 kHz — will be resampled to 48 kHz
        data = np.ones(12000, dtype=np.float32) * 0.5  # 0.5s at 24kHz
        sf.write(str(synth_dir / "24k.wav"), data, 24000)

        track = _build_speech_track(segments, synth_dir, total_duration=2.0, sample_rate=48000)
        # After resampling 0.5s @ 24kHz → ~24000 samples @ 48kHz
        # Check that data was placed (non-zero)
        segment_region = track[0:24000]
        assert np.any(segment_region != 0.0)


# ── FFmpeg assembly tests ──────────────────────────────────────────────────


class TestAssembleVideo:
    def test_softsub_mp4(self, tmp_path: Path) -> None:
        """Softsub adds subtitle stream with mov_text codec."""
        srt = tmp_path / "subs.srt"
        srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHi\n")

        with patch("yt_dbl.pipeline.assemble.run_ffmpeg") as mock_ff:
            _assemble_video(
                video_path=tmp_path / "v.mp4",
                speech_path=tmp_path / "s.wav",
                background_path=tmp_path / "bg.wav",
                output_path=tmp_path / "out.mp4",
                background_volume=0.15,
                subtitle_path=srt,
                subtitle_mode="softsub",
                output_format="mp4",
            )
            mock_ff.assert_called_once()
            args = mock_ff.call_args[0][0]

            # 4 inputs (video, speech, bg, srt)
            assert args.count("-i") == 4
            # Video copied (not re-encoded)
            assert "-c:v" in args
            idx = args.index("-c:v")
            assert args[idx + 1] == "copy"
            # Subtitle codec
            assert "mov_text" in args
            # Audio filter with ducking + limiter
            fc_idx = args.index("-filter_complex")
            fc = args[fc_idx + 1]
            assert "volume=0.15" in fc
            assert "sidechaincompress" in fc
            assert "alimiter" in fc
            assert "amix" in fc

    def test_softsub_mkv(self, tmp_path: Path) -> None:
        """MKV uses srt codec for subtitles."""
        srt = tmp_path / "subs.srt"
        srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHi\n")

        with patch("yt_dbl.pipeline.assemble.run_ffmpeg") as mock_ff:
            _assemble_video(
                video_path=tmp_path / "v.mkv",
                speech_path=tmp_path / "s.wav",
                background_path=tmp_path / "bg.wav",
                output_path=tmp_path / "out.mkv",
                background_volume=0.2,
                subtitle_path=srt,
                subtitle_mode="softsub",
                output_format="mkv",
            )
            args = mock_ff.call_args[0][0]
            # MKV subtitle codec is "srt"
            assert "-c:s" in args
            idx = args.index("-c:s")
            assert args[idx + 1] == "srt"

    def test_hardsub(self, tmp_path: Path) -> None:
        """Hardsub re-encodes video with subtitle burn-in."""
        srt = tmp_path / "subs.srt"
        srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHi\n")

        with patch("yt_dbl.pipeline.assemble.run_ffmpeg") as mock_ff:
            _assemble_video(
                video_path=tmp_path / "v.mp4",
                speech_path=tmp_path / "s.wav",
                background_path=tmp_path / "bg.wav",
                output_path=tmp_path / "out.mp4",
                background_volume=0.15,
                subtitle_path=srt,
                subtitle_mode="hardsub",
                output_format="mp4",
            )
            args = mock_ff.call_args[0][0]
            # Video re-encoded with libx264
            idx = args.index("-c:v")
            assert args[idx + 1] == "libx264"
            # Filter complex includes subtitles filter
            fc_idx = args.index("-filter_complex")
            fc = args[fc_idx + 1]
            assert "subtitles=" in fc

    def test_no_subtitles(self, tmp_path: Path) -> None:
        """When subtitle_mode=none, no subtitle input or mapping."""
        with patch("yt_dbl.pipeline.assemble.run_ffmpeg") as mock_ff:
            _assemble_video(
                video_path=tmp_path / "v.mp4",
                speech_path=tmp_path / "s.wav",
                background_path=tmp_path / "bg.wav",
                output_path=tmp_path / "out.mp4",
                background_volume=0.15,
                subtitle_path=None,
                subtitle_mode="none",
            )
            args = mock_ff.call_args[0][0]
            # Only 3 inputs (no subtitle)
            assert args.count("-i") == 3
            assert "-c:s" not in args

    def test_ducking_disabled(self, tmp_path: Path) -> None:
        """With ducking off, no sidechaincompress but limiter is still present."""
        with patch("yt_dbl.pipeline.assemble.run_ffmpeg") as mock_ff:
            _assemble_video(
                video_path=tmp_path / "v.mp4",
                speech_path=tmp_path / "s.wav",
                background_path=tmp_path / "bg.wav",
                output_path=tmp_path / "out.mp4",
                background_volume=0.15,
                background_ducking=False,
            )
            args = mock_ff.call_args[0][0]
            fc_idx = args.index("-filter_complex")
            fc = args[fc_idx + 1]
            assert "sidechaincompress" not in fc
            assert "alimiter" in fc
            assert "amix" in fc

    def test_missing_subtitle_file(self, tmp_path: Path) -> None:
        """Falls back gracefully if subtitle file doesn't exist."""
        with patch("yt_dbl.pipeline.assemble.run_ffmpeg") as mock_ff:
            _assemble_video(
                video_path=tmp_path / "v.mp4",
                speech_path=tmp_path / "s.wav",
                background_path=tmp_path / "bg.wav",
                output_path=tmp_path / "out.mp4",
                background_volume=0.15,
                subtitle_path=tmp_path / "nonexistent.srt",
                subtitle_mode="softsub",
            )
            args = mock_ff.call_args[0][0]
            assert args.count("-i") == 3  # no subtitle input

    def test_aac_320k(self, tmp_path: Path) -> None:
        """Audio is encoded as AAC 320kbps."""
        with patch("yt_dbl.pipeline.assemble.run_ffmpeg") as mock_ff:
            _assemble_video(
                video_path=tmp_path / "v.mp4",
                speech_path=tmp_path / "s.wav",
                background_path=tmp_path / "bg.wav",
                output_path=tmp_path / "out.mp4",
                background_volume=0.15,
            )
            args = mock_ff.call_args[0][0]
            assert "-c:a" in args
            idx = args.index("-c:a")
            assert args[idx + 1] == "aac"
            assert "-b:a" in args
            idx = args.index("-b:a")
            assert args[idx + 1] == "320k"


# ── Full run tests (mocked ffmpeg) ─────────────────────────────────────────


class TestAssembleStepRun:
    def test_run_success(self, tmp_path: Path) -> None:
        step, cfg, state = _make_step(tmp_path)

        with patch(
            "yt_dbl.pipeline.assemble.run_ffmpeg",
            return_value=subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
        ):
            # Create the output file (ffmpeg is mocked, so simulate it)
            job_dir = step.step_dir.parent
            output_path = job_dir / f"result.{cfg.output_format}"

            # Patch run_ffmpeg to also create the output file
            def fake_assemble(*args: object, **kwargs: object) -> Path:
                output_path.write_bytes(b"fake-result")
                return output_path

            with patch("yt_dbl.pipeline.assemble._assemble_video", side_effect=fake_assemble):
                state = step.run(state)

        result = state.get_step(StepName.ASSEMBLE)
        assert "result" in result.outputs
        assert result.outputs["result"] == "result.mp4"

        # Speech track should have been written
        speech_path = step.step_dir / SPEECH_TRACK_FILE
        assert speech_path.exists()

    def test_run_creates_speech_track(self, tmp_path: Path) -> None:
        """Speech track file is created from segments."""
        step, _, state = _make_step(tmp_path)

        with patch("yt_dbl.pipeline.assemble.run_ffmpeg"):
            state = step.run(state)

        speech_path = step.step_dir / SPEECH_TRACK_FILE
        assert speech_path.exists()

        data, sr = sf.read(str(speech_path))
        assert sr == 48000
        # Duration should be ~13s (12s video + 1s safety)
        assert len(data) == int(12.0 * 48000) + 48000

    def test_run_idempotent(self, tmp_path: Path) -> None:
        """If result.mp4 already exists, skip assembly."""
        step, cfg, state = _make_step(tmp_path)
        job_dir = step.step_dir.parent
        output_path = job_dir / f"result.{cfg.output_format}"
        output_path.write_bytes(b"existing-result")

        # Should NOT call ffmpeg
        with patch("yt_dbl.pipeline.assemble.run_ffmpeg") as mock_ff:
            state = step.run(state)
            mock_ff.assert_not_called()

        assert state.get_step(StepName.ASSEMBLE).outputs["result"] == "result.mp4"

    def test_run_with_subtitle_output(self, tmp_path: Path) -> None:
        """Subtitle path is propagated to outputs."""
        step, _, state = _make_step(tmp_path)

        with patch("yt_dbl.pipeline.assemble.run_ffmpeg"):
            state = step.run(state)

        result = state.get_step(StepName.ASSEMBLE)
        assert "subtitles" in result.outputs

    def test_run_mkv_format(self, tmp_path: Path) -> None:
        """Output file uses correct extension for mkv format."""
        cfg = Settings(work_dir=tmp_path / "work", output_format="mkv")
        step_dir = cfg.step_dir("test123", "06_assemble")
        step = AssembleStep(settings=cfg, work_dir=step_dir)

        _, _, state = _make_step(tmp_path)
        # Recreate synth files in the new work_dir
        synth_dir = cfg.step_dir("test123", "05_synthesize")
        for seg in state.segments:
            _write_wav(synth_dir / seg.synth_path, duration_sec=seg.duration)

        # Recreate other step files
        dl_dir = cfg.step_dir("test123", "01_download")
        (dl_dir / "video.mp4").write_bytes(b"fake-video")
        sep_dir = cfg.step_dir("test123", "02_separate")
        _write_wav(sep_dir / "background.wav", duration_sec=12.0)
        trans_dir = cfg.step_dir("test123", "04_translate")
        (trans_dir / "subtitles.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nHi\n")

        with patch("yt_dbl.pipeline.assemble.run_ffmpeg"):
            state = step.run(state)

        assert state.get_step(StepName.ASSEMBLE).outputs["result"] == "result.mkv"


# ── Duration resolution tests ──────────────────────────────────────────────


class TestGetTotalDuration:
    def test_from_meta(self, tmp_path: Path) -> None:
        step, _, state = _make_step(tmp_path)
        assert state.meta is not None
        video_path = tmp_path / "v.mp4"
        dur = step._get_total_duration(state, video_path)
        assert dur == 12.0

    def test_from_ffprobe_fallback(self, tmp_path: Path) -> None:
        step, _, state = _make_step(tmp_path)
        state.meta = None
        video_path = tmp_path / "v.mp4"

        with patch("yt_dbl.pipeline.assemble.get_audio_duration", return_value=15.0):
            dur = step._get_total_duration(state, video_path)
        assert dur == 15.0

    def test_from_segments_fallback(self, tmp_path: Path) -> None:
        step, _, state = _make_step(tmp_path)
        state.meta = None
        video_path = tmp_path / "v.mp4"

        with patch(
            "yt_dbl.pipeline.assemble.get_audio_duration",
            side_effect=RuntimeError("no ffprobe"),
        ):
            dur = step._get_total_duration(state, video_path)
        # Last segment ends at 10.0 → +1.0 buffer = 11.0
        assert dur == 11.0

    def test_default_fallback(self, tmp_path: Path) -> None:
        step, _, state = _make_step(tmp_path)
        state.meta = None
        state.segments = []
        video_path = tmp_path / "v.mp4"

        with patch(
            "yt_dbl.pipeline.assemble.get_audio_duration",
            side_effect=RuntimeError("fail"),
        ):
            dur = step._get_total_duration(state, video_path)
        assert dur == 60.0


# ── Config tests ────────────────────────────────────────────────────────────


class TestAssembleConfig:
    def test_default_subtitle_mode(self) -> None:
        cfg = Settings(_env_file=None)  # type: ignore[call-arg]
        assert cfg.subtitle_mode == "softsub"

    def test_subtitle_mode_via_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("YT_DBL_SUBTITLE_MODE", "hardsub")
        cfg = Settings(_env_file=None)  # type: ignore[call-arg]
        assert cfg.subtitle_mode == "hardsub"

    def test_subtitle_mode_none_via_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("YT_DBL_SUBTITLE_MODE", "none")
        cfg = Settings(_env_file=None)  # type: ignore[call-arg]
        assert cfg.subtitle_mode == "none"
