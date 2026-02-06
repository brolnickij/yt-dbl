"""Tests for yt_dbl.pipeline.synthesize — TTS + voice cloning (mocked)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from yt_dbl.config import Settings
from yt_dbl.pipeline.base import StepValidationError
from yt_dbl.pipeline.synthesize import (
    SYNTH_META_FILE,
    SynthesizeStep,
    _find_ref_text_for_speaker,
)
from yt_dbl.schemas import STEP_DIRS, PipelineState, Segment, Speaker, StepName, StepStatus, Word

# ── Helpers ─────────────────────────────────────────────────────────────────


def _ffmpeg_touch(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
    """Mock run_ffmpeg that creates the output file so downstream steps find it."""
    cmd_args: list[str] = args[0] if args else kwargs.get("args", [])
    # Last positional arg is the output path (unless it's /dev/null)
    if cmd_args:
        candidate = cmd_args[-1]
        if candidate != "/dev/null" and not candidate.startswith("-"):
            Path(candidate).parent.mkdir(parents=True, exist_ok=True)
            Path(candidate).write_bytes(b"fake-audio")
    return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")


def _fake_postprocess(input_path: Path, output_path: Path, **_kwargs: Any) -> Path:
    """Mock postprocess_segment that creates the output file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(b"fake-audio")
    return output_path


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
    step_dir = cfg.step_dir("test123", STEP_DIRS[StepName.SYNTHESIZE])
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
    sep_dir = cfg.step_dir("test123", STEP_DIRS[StepName.SEPARATE])
    (sep_dir / "vocals.wav").write_bytes(b"fake-vocals")

    return step, cfg, state


def _fake_run_tts(
    _self: object, _model: object, _text: object, _ref: object, _ref_text: object, _lang: object
) -> Any:
    """Return a plain numpy array instead of mlx array."""
    import numpy as np

    return np.zeros(24000, dtype=np.float32)


def _fake_save_wav(_self: object, _audio: object, path: Path, _sr: object) -> None:
    """Write a minimal WAV file without mlx dependency."""
    import numpy as np
    import soundfile as sf

    sf.write(str(path), np.zeros(24000, dtype=np.float32), 24000)


# ── Validation tests ────────────────────────────────────────────────────────


class TestSynthesizeStepValidation:
    def test_validate_no_segments(self, tmp_path: Path) -> None:
        step, _, state = _make_step(tmp_path)
        state.segments = []
        with pytest.raises(StepValidationError, match="No segments"):
            step.validate_inputs(state)

    def test_validate_no_translated_text(self, tmp_path: Path) -> None:
        step, _, state = _make_step(tmp_path)
        for seg in state.segments:
            seg.translated_text = ""
        with pytest.raises(StepValidationError, match="no translated text"):
            step.validate_inputs(state)

    def test_validate_no_vocals(self, tmp_path: Path) -> None:
        step, _, state = _make_step(tmp_path)
        state.get_step(StepName.SEPARATE).outputs = {}
        with pytest.raises(StepValidationError, match="No vocals"):
            step.validate_inputs(state)

    def test_validate_no_speakers(self, tmp_path: Path) -> None:
        step, _, state = _make_step(tmp_path)
        state.speakers = []
        with pytest.raises(StepValidationError, match="No speakers"):
            step.validate_inputs(state)

    def test_validate_ok(self, tmp_path: Path) -> None:
        step, _, state = _make_step(tmp_path)
        step.validate_inputs(state)  # should not raise


# ── Voice reference text tests ──────────────────────────────────────────────


class TestFindRefTextForSpeaker:
    def test_exact_match(self) -> None:
        segments = _make_segments()
        speaker = Speaker(id="SPEAKER_00", reference_start=0.0, reference_end=3.5)
        text = _find_ref_text_for_speaker(segments, speaker)
        assert text == "Hello, welcome to this video."

    def test_fallback(self) -> None:
        segments = _make_segments()
        speaker = Speaker(id="SPEAKER_00", reference_start=99.0, reference_end=100.0)
        text = _find_ref_text_for_speaker(segments, speaker)
        # Falls back to first segment of that speaker
        assert text == "Hello, welcome to this video."

    def test_unknown_speaker(self) -> None:
        segments = _make_segments()
        speaker = Speaker(id="SPEAKER_99")
        assert _find_ref_text_for_speaker(segments, speaker) == ""


# ── Persistence tests ─────────────────────────────────────────────────────────


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

        mock_model = MagicMock()

        with (
            patch(
                "yt_dbl.pipeline.synthesize.SynthesizeStep._load_tts_model",
                return_value=mock_model,
            ),
            patch(
                "yt_dbl.pipeline.synthesize.SynthesizeStep._run_tts",
                _fake_run_tts,
            ),
            patch(
                "yt_dbl.pipeline.synthesize.SynthesizeStep._save_wav",
                _fake_save_wav,
            ),
            patch(
                "yt_dbl.utils.audio_processing.run_ffmpeg",
                side_effect=_ffmpeg_touch,
            ),
            patch(
                "yt_dbl.pipeline.synthesize.postprocess_segment",
                side_effect=_fake_postprocess,
            ),
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

        mock_model = MagicMock()

        with (
            patch(
                "yt_dbl.pipeline.synthesize.SynthesizeStep._load_tts_model",
                return_value=mock_model,
            ),
            patch(
                "yt_dbl.pipeline.synthesize.SynthesizeStep._run_tts",
                _fake_run_tts,
            ),
            patch(
                "yt_dbl.pipeline.synthesize.SynthesizeStep._save_wav",
                _fake_save_wav,
            ),
            patch(
                "yt_dbl.utils.audio_processing.run_ffmpeg",
                side_effect=_ffmpeg_touch,
            ),
            patch(
                "yt_dbl.pipeline.synthesize.postprocess_segment",
                side_effect=_fake_postprocess,
            ),
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
        cfg = Settings()
        assert cfg.tts_model == "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"
        assert cfg.tts_sample_rate == 24000
        assert cfg.tts_temperature == 0.9
        assert cfg.tts_top_k == 50
        assert cfg.tts_top_p == 1.0
        assert cfg.tts_repetition_penalty == 1.05

    def test_custom_model_via_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("YT_DBL_TTS_MODEL", "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16")
        cfg = Settings()
        assert cfg.tts_model == "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16"


# ── Speaker helper tests
