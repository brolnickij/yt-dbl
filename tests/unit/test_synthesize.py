"""Tests for yt_dbl.pipeline.synthesize — TTS + voice cloning (mocked)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from yt_dbl.config import Settings
from yt_dbl.pipeline.synthesize import (
    SYNTH_META_FILE,
    SynthesizeStep,
    _extract_voice_reference,
    _find_ref_text_for_speaker,
    _normalize_loudness,
    _speed_up_audio,
)
from yt_dbl.schemas import PipelineState, Segment, Speaker, StepName, StepStatus, Word

if TYPE_CHECKING:
    from pathlib import Path


# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_segments() -> list[Segment]:
    return [
        Segment(
            id=0,
            text="Hello, welcome to this video.",
            start=0.0,
            end=3.5,
            speaker="SPEAKER_00",
            language="en",
            translated_text="Privet, dobro pozhalovat.",
            words=[Word(text="Hello,", start=0.0, end=0.5)],
        ),
        Segment(
            id=1,
            text="Today we talk about something.",
            start=4.0,
            end=8.0,
            speaker="SPEAKER_00",
            language="en",
            translated_text="Segodnya pogovorim.",
        ),
        Segment(
            id=2,
            text="That sounds great!",
            start=9.0,
            end=11.0,
            speaker="SPEAKER_01",
            language="en",
            translated_text="Zvuchit otlichno!",
        ),
    ]


def _make_speakers() -> list[Speaker]:
    return [
        Speaker(
            id="SPEAKER_00",
            reference_start=0.0,
            reference_end=3.5,
            total_duration=7.5,
        ),
        Speaker(
            id="SPEAKER_01",
            reference_start=9.0,
            reference_end=11.0,
            total_duration=2.0,
        ),
    ]


def _make_step(tmp_path: Path) -> tuple[SynthesizeStep, Settings, PipelineState]:
    cfg = Settings(work_dir=tmp_path / "work")
    step_dir = cfg.step_dir("test123", "05_synthesize")
    step = SynthesizeStep(settings=cfg, work_dir=step_dir)

    state = PipelineState(video_id="test123", url="https://example.com")
    state.segments = _make_segments()
    state.speakers = _make_speakers()

    # Prefill earlier steps
    sep = state.get_step(StepName.SEPARATE)
    sep.status = StepStatus.COMPLETED
    sep.outputs = {"vocals": "vocals.wav", "background": "background.wav"}

    trans = state.get_step(StepName.TRANSLATE)
    trans.status = StepStatus.COMPLETED

    # Create fake vocals file
    sep_dir = cfg.step_dir("test123", "02_separate")
    (sep_dir / "vocals.wav").write_bytes(b"fake-vocals")

    return step, cfg, state


def _fake_tts_result() -> MagicMock:
    """Create a fake TTS generation result."""
    import numpy as np

    result = MagicMock()
    result.audio = np.zeros(12000, dtype=np.float32)  # 1 second at 12kHz
    result.sample_rate = 12000
    return result


# ── Validation tests ────────────────────────────────────────────────────────


class TestSynthesizeStepValidation:
    def test_validate_no_segments(self, tmp_path: Path) -> None:
        step, _, state = _make_step(tmp_path)
        state.segments = []
        with pytest.raises(ValueError, match="No segments"):
            step.validate_inputs(state)

    def test_validate_no_translated_text(self, tmp_path: Path) -> None:
        step, _, state = _make_step(tmp_path)
        for seg in state.segments:
            seg.translated_text = ""
        with pytest.raises(ValueError, match="no translated text"):
            step.validate_inputs(state)

    def test_validate_no_vocals(self, tmp_path: Path) -> None:
        step, _, state = _make_step(tmp_path)
        state.get_step(StepName.SEPARATE).outputs = {}
        with pytest.raises(ValueError, match="No vocals"):
            step.validate_inputs(state)

    def test_validate_no_speakers(self, tmp_path: Path) -> None:
        step, _, state = _make_step(tmp_path)
        state.speakers = []
        with pytest.raises(ValueError, match="No speakers"):
            step.validate_inputs(state)

    def test_validate_ok(self, tmp_path: Path) -> None:
        step, _, state = _make_step(tmp_path)
        step.validate_inputs(state)  # should not raise


# ── Voice reference tests ───────────────────────────────────────────────────


class TestVoiceReference:
    def test_extract_calls_ffmpeg(self, tmp_path: Path) -> None:
        speaker = Speaker(id="SPEAKER_00", reference_start=1.0, reference_end=5.0)
        output = tmp_path / "ref.wav"

        with patch("yt_dbl.pipeline.synthesize.run_ffmpeg") as mock_ff:
            _extract_voice_reference(tmp_path / "vocals.wav", speaker, output, target_duration=7.0)
            mock_ff.assert_called_once()
            args = mock_ff.call_args[0][0]
            assert "-ss" in args
            assert "1.0" in args
            assert "-t" in args
            assert "4.0" in args  # min(5.0-1.0, 7.0) = 4.0

    def test_find_ref_text_exact_match(self) -> None:
        segments = _make_segments()
        speaker = Speaker(id="SPEAKER_00", reference_start=0.0, reference_end=3.5)
        text = _find_ref_text_for_speaker(segments, speaker)
        assert text == "Hello, welcome to this video."

    def test_find_ref_text_fallback(self) -> None:
        segments = _make_segments()
        speaker = Speaker(id="SPEAKER_00", reference_start=99.0, reference_end=100.0)
        text = _find_ref_text_for_speaker(segments, speaker)
        # Falls back to first segment of that speaker
        assert text == "Hello, welcome to this video."

    def test_find_ref_text_unknown_speaker(self) -> None:
        segments = _make_segments()
        speaker = Speaker(id="SPEAKER_99")
        assert _find_ref_text_for_speaker(segments, speaker) == ""


# ── Speed adjustment tests ──────────────────────────────────────────────────


class TestSpeedAdjust:
    def test_speed_up_calls_ffmpeg(self, tmp_path: Path) -> None:
        with patch("yt_dbl.pipeline.synthesize.run_ffmpeg") as mock_ff:
            _speed_up_audio(tmp_path / "in.wav", tmp_path / "out.wav", 1.3)
            mock_ff.assert_called_once()
            args = mock_ff.call_args[0][0]
            assert "atempo=1.3000" in args[-1] or "atempo=1.3" in str(args)

    def test_normalize_calls_ffmpeg(self, tmp_path: Path) -> None:
        with patch("yt_dbl.pipeline.synthesize.run_ffmpeg") as mock_ff:
            _normalize_loudness(tmp_path / "in.wav", tmp_path / "out.wav")
            mock_ff.assert_called_once()
            args = mock_ff.call_args[0][0]
            assert any("loudnorm" in a for a in args)


# ── Persistence tests ───────────────────────────────────────────────────────


class TestPersistence:
    def test_save_and_load_meta(self, tmp_path: Path) -> None:
        step, _, state = _make_step(tmp_path)

        # Simulate synthesized state
        for seg in state.segments:
            seg.synth_path = f"segment_{seg.id:04d}.wav"
            seg.synth_speed_factor = 1.2
        state.speakers[0].reference_path = "ref_SPEAKER_00.wav"

        meta_path = step.step_dir / SYNTH_META_FILE
        step._save_meta(state, meta_path)

        data = json.loads(meta_path.read_text())
        assert len(data["segments"]) == 3
        assert data["segments"][0]["synth_path"] == "segment_0000.wav"
        assert data["speakers"][0]["reference_path"] == "ref_SPEAKER_00.wav"

    def test_load_cached(self, tmp_path: Path) -> None:
        step, _, state = _make_step(tmp_path)

        # Write meta
        for seg in state.segments:
            seg.synth_path = f"segment_{seg.id:04d}.wav"
        step._save_meta(state, step.step_dir / SYNTH_META_FILE)

        # Reset synth_path and reload
        for seg in state.segments:
            seg.synth_path = ""

        state = step._load_cached(state, step.step_dir / SYNTH_META_FILE)

        assert state.segments[0].synth_path == "segment_0000.wav"
        assert state.segments[2].synth_path == "segment_0002.wav"
        result = state.get_step(StepName.SYNTHESIZE)
        assert "seg_0" in result.outputs
        assert "meta" in result.outputs


# ── Full run tests (mocked TTS + ffmpeg) ───────────────────────────────────


class TestSynthesizeStepRun:
    def test_run_success(self, tmp_path: Path) -> None:
        step, _, state = _make_step(tmp_path)
        fake_result = _fake_tts_result()

        mock_model = MagicMock()
        mock_model.generate.return_value = [fake_result]

        with (
            patch(
                "yt_dbl.pipeline.synthesize.SynthesizeStep._load_tts_model",
                return_value=mock_model,
            ),
            patch("yt_dbl.pipeline.synthesize.run_ffmpeg"),
            patch(
                "yt_dbl.pipeline.synthesize.get_audio_duration",
                return_value=1.0,  # shorter than original → no speedup
            ),
        ):
            state = step.run(state)

        result = state.get_step(StepName.SYNTHESIZE)
        assert "seg_0" in result.outputs
        assert "meta" in result.outputs
        assert (step.step_dir / SYNTH_META_FILE).exists()

        # All segments should have synth_path set
        for seg in state.segments:
            assert seg.synth_path != ""

    def test_run_with_speedup(self, tmp_path: Path) -> None:
        step, _, state = _make_step(tmp_path)
        fake_result = _fake_tts_result()

        mock_model = MagicMock()
        mock_model.generate.return_value = [fake_result]

        with (
            patch(
                "yt_dbl.pipeline.synthesize.SynthesizeStep._load_tts_model",
                return_value=mock_model,
            ),
            patch("yt_dbl.pipeline.synthesize.run_ffmpeg"),
            patch(
                "yt_dbl.pipeline.synthesize.get_audio_duration",
                return_value=10.0,  # longer than original → speedup needed
            ),
        ):
            state = step.run(state)

        # At least one segment should have speed factor > 1
        sped = [seg for seg in state.segments if seg.synth_speed_factor > 1.01]
        assert len(sped) > 0

    def test_run_idempotent(self, tmp_path: Path) -> None:
        step, _, state = _make_step(tmp_path)

        # Pre-create meta
        for seg in state.segments:
            seg.synth_path = f"segment_{seg.id:04d}.wav"
        step._save_meta(state, step.step_dir / SYNTH_META_FILE)

        # Reset and run — should load from cache
        for seg in state.segments:
            seg.synth_path = ""

        state = step.run(state)

        assert state.segments[0].synth_path == "segment_0000.wav"
        assert state.get_step(StepName.SYNTHESIZE).outputs["meta"] == SYNTH_META_FILE


# ── Config tests ────────────────────────────────────────────────────────────


class TestSynthesisConfig:
    def test_default_tts_model(self) -> None:
        cfg = Settings(_env_file=None)  # type: ignore[call-arg]
        assert cfg.tts_model == "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"
        assert cfg.tts_sample_rate == 12000
        assert cfg.tts_temperature == 0.9

    def test_custom_model_via_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("YT_DBL_TTS_MODEL", "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16")
        cfg = Settings(_env_file=None)  # type: ignore[call-arg]
        assert cfg.tts_model == "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16"


# ── Speaker helper tests ───────────────────────────────────────────────────


class TestSpeakerById:
    def test_finds_existing(self, tmp_path: Path) -> None:
        step, _, state = _make_step(tmp_path)
        s = step._speaker_by_id(state, "SPEAKER_00")
        assert s.id == "SPEAKER_00"
        assert s.total_duration == 7.5

    def test_returns_default_for_unknown(self, tmp_path: Path) -> None:
        step, _, state = _make_step(tmp_path)
        s = step._speaker_by_id(state, "SPEAKER_99")
        assert s.id == "SPEAKER_99"
        assert s.total_duration == 0.0
