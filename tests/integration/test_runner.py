"""Tests for pipeline runner — checkpointing and resume."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from tests.conftest import prefill_download
from yt_dbl.config import Settings
from yt_dbl.pipeline.runner import PipelineRunner, load_state, save_state
from yt_dbl.schemas import PipelineState, Segment, StepName, StepStatus


def _fake_separation_factory(sep_dir: Path) -> Any:
    """Return a side_effect callable that creates fake separation outputs."""

    def _fake(audio_path: Path) -> None:
        (sep_dir / "vocals.wav").write_bytes(b"fake-vocals")
        (sep_dir / "background.wav").write_bytes(b"fake-background")

    return _fake


def _fake_transcription_factory() -> dict[str, Any]:
    """Return mock side_effects for _run_asr and _run_alignment."""
    from yt_dbl.schemas import Segment, Word

    fake_asr_segments = [
        {"start": 0.0, "end": 3.0, "speaker_id": 0, "text": "Hello world."},
        {"start": 4.0, "end": 6.0, "speaker_id": 1, "text": "Goodbye."},
    ]

    fake_segments = [
        Segment(
            id=0,
            text="Hello world.",
            start=0.0,
            end=3.0,
            speaker="SPEAKER_00",
            language="en",
            words=[Word(text="Hello", start=0.0, end=1.5), Word(text="world.", start=1.5, end=3.0)],
        ),
        Segment(
            id=1,
            text="Goodbye.",
            start=4.0,
            end=6.0,
            speaker="SPEAKER_01",
            language="en",
            words=[Word(text="Goodbye.", start=4.0, end=6.0)],
        ),
    ]

    def _fake_asr(vocals_path: Path) -> list[dict[str, Any]]:
        return fake_asr_segments

    def _fake_alignment(
        vocals_path: Path,
        raw_segments: list[dict[str, Any]],
    ) -> list[Segment]:
        return fake_segments

    return {
        "asr": _fake_asr,
        "alignment": _fake_alignment,
    }


def _fake_translation(
    segments: list[Segment],
    target_language: str,
) -> dict[int, str]:
    """Fake translate that returns deterministic translations."""
    return {seg.id: f"[{target_language}] {seg.text}" for seg in segments}


def _fake_tts_model() -> Any:
    """Return a fake TTS model whose generate() returns dummy audio."""
    import numpy as np

    model = MagicMock()
    result = MagicMock()
    result.audio = np.zeros(12000, dtype=np.float32)
    result.sample_rate = 12000
    model.generate.return_value = [result]
    return model


def _pipeline_patches(sep_dir: Path) -> Any:
    """Context manager stack for all pipeline mocks (separate+transcribe+translate+synthesize)."""
    from contextlib import ExitStack

    fakes = _fake_transcription_factory()

    stack = ExitStack()
    stack.enter_context(
        patch(
            "yt_dbl.pipeline.separate.SeparateStep._run_separation",
            side_effect=_fake_separation_factory(sep_dir),
        )
    )
    stack.enter_context(
        patch(
            "yt_dbl.pipeline.transcribe.TranscribeStep._run_asr",
            side_effect=fakes["asr"],
        )
    )
    stack.enter_context(
        patch(
            "yt_dbl.pipeline.transcribe.TranscribeStep._run_alignment",
            side_effect=fakes["alignment"],
        )
    )
    stack.enter_context(
        patch(
            "yt_dbl.pipeline.translate.TranslateStep._translate",
            side_effect=_fake_translation,
        )
    )
    # Synthesize mocks: skip real TTS + ffmpeg
    stack.enter_context(
        patch(
            "yt_dbl.pipeline.synthesize.SynthesizeStep._load_tts_model",
            return_value=_fake_tts_model(),
        )
    )
    stack.enter_context(
        patch("yt_dbl.pipeline.synthesize.run_ffmpeg"),
    )
    stack.enter_context(
        patch(
            "yt_dbl.pipeline.synthesize.get_audio_duration",
            return_value=1.0,
        ),
    )
    stack.enter_context(
        patch("yt_dbl.pipeline.synthesize.SynthesizeStep._save_wav"),
    )
    return stack


class TestCheckpoints:
    def test_save_and_load(self, tmp_path: Path) -> None:
        cfg = Settings(work_dir=tmp_path / "work")
        state = PipelineState(video_id="test123", url="https://example.com", target_language="ru")
        state.get_step(StepName.DOWNLOAD).status = StepStatus.COMPLETED

        save_state(state, cfg)
        loaded = load_state(cfg, "test123")

        assert loaded is not None
        assert loaded.video_id == "test123"
        assert loaded.get_step(StepName.DOWNLOAD).status == StepStatus.COMPLETED

    def test_load_nonexistent(self, tmp_path: Path) -> None:
        cfg = Settings(work_dir=tmp_path / "work")
        assert load_state(cfg, "nonexistent") is None

    def test_state_file_is_valid_json(self, tmp_path: Path) -> None:
        cfg = Settings(work_dir=tmp_path / "work")
        state = PipelineState(video_id="test123")
        path = save_state(state, cfg)

        data = json.loads(path.read_text())
        assert data["video_id"] == "test123"


class TestPipelineRunner:
    def test_pipeline_from_separate(self, tmp_path: Path) -> None:
        """Run pipeline skipping download (which needs real yt-dlp)."""
        cfg = Settings(work_dir=tmp_path / "work", anthropic_api_key="sk-test")
        state = PipelineState(
            video_id="test123",
            url="https://youtube.com/watch?v=test123",
            target_language="ru",
        )
        state = prefill_download(state, cfg)
        save_state(state, cfg)

        sep_dir = cfg.step_dir("test123", "02_separate")

        runner = PipelineRunner(cfg)
        with _pipeline_patches(sep_dir):
            state = runner.run(state)

        assert state.next_step is None
        for step_name in [
            StepName.SEPARATE,
            StepName.TRANSCRIBE,
            StepName.TRANSLATE,
            StepName.SYNTHESIZE,
            StepName.ASSEMBLE,
        ]:
            assert state.get_step(step_name).status == StepStatus.COMPLETED

    def test_resume_skips_completed(self, tmp_path: Path) -> None:
        cfg = Settings(work_dir=tmp_path / "work", anthropic_api_key="sk-test")
        state = PipelineState(video_id="test123", url="https://example.com")
        state = prefill_download(state, cfg)

        sep_dir = cfg.step_dir("test123", "02_separate")

        runner = PipelineRunner(cfg)
        with _pipeline_patches(sep_dir):
            state = runner.run(state)

        # Mark last 3 as pending again
        for name in [StepName.TRANSLATE, StepName.SYNTHESIZE, StepName.ASSEMBLE]:
            state.get_step(name).status = StepStatus.PENDING
        save_state(state, cfg)

        loaded = load_state(cfg, "test123")
        assert loaded is not None
        assert loaded.next_step == StepName.TRANSLATE

    def test_from_step(self, tmp_path: Path) -> None:
        cfg = Settings(work_dir=tmp_path / "work", anthropic_api_key="sk-test")
        state = PipelineState(video_id="test123", url="https://example.com")
        state = prefill_download(state, cfg)

        sep_dir = cfg.step_dir("test123", "02_separate")

        runner = PipelineRunner(cfg)
        with _pipeline_patches(sep_dir):
            state = runner.run(state)
            state = runner.run(state, from_step=StepName.TRANSLATE)

        assert state.get_step(StepName.TRANSLATE).status == StepStatus.COMPLETED

    def test_step_failure_stops_pipeline(self, tmp_path: Path) -> None:
        """When a step raises, its status becomes FAILED and pipeline stops."""
        cfg = Settings(work_dir=tmp_path / "work")
        state = PipelineState(video_id="test123", url="https://example.com")
        state = prefill_download(state, cfg)

        runner = PipelineRunner(cfg)

        # Patch SeparateStep.run to raise
        with patch(
            "yt_dbl.pipeline.separate.SeparateStep.run",
            side_effect=RuntimeError("boom"),
        ):
            state = runner.run(state)

        assert state.get_step(StepName.SEPARATE).status == StepStatus.FAILED
        assert "boom" in state.get_step(StepName.SEPARATE).error
        # Steps after failure should still be pending
        assert state.get_step(StepName.TRANSCRIBE).status == StepStatus.PENDING

    def test_checkpoint_saved_on_success(self, tmp_path: Path) -> None:
        """After each step completes, state is persisted to disk."""
        cfg = Settings(work_dir=tmp_path / "work", anthropic_api_key="sk-test")
        state = PipelineState(video_id="test123", url="https://example.com")
        state = prefill_download(state, cfg)

        sep_dir = cfg.step_dir("test123", "02_separate")

        runner = PipelineRunner(cfg)
        with _pipeline_patches(sep_dir):
            state = runner.run(state)

        loaded = load_state(cfg, "test123")
        assert loaded is not None
        assert loaded.get_step(StepName.ASSEMBLE).status == StepStatus.COMPLETED

    def test_checkpoint_saved_on_failure(self, tmp_path: Path) -> None:
        """Failed state should also be persisted."""
        cfg = Settings(work_dir=tmp_path / "work")
        state = PipelineState(video_id="test123", url="https://example.com")
        state = prefill_download(state, cfg)

        runner = PipelineRunner(cfg)

        with patch(
            "yt_dbl.pipeline.separate.SeparateStep.run",
            side_effect=RuntimeError("disk full"),
        ):
            state = runner.run(state)

        loaded = load_state(cfg, "test123")
        assert loaded is not None
        assert loaded.get_step(StepName.SEPARATE).status == StepStatus.FAILED
        assert "disk full" in loaded.get_step(StepName.SEPARATE).error

    def test_step_timing_recorded(self, tmp_path: Path) -> None:
        """Completed steps record timing metadata."""
        cfg = Settings(work_dir=tmp_path / "work", anthropic_api_key="sk-test")
        state = PipelineState(video_id="test123", url="https://example.com")
        state = prefill_download(state, cfg)

        sep_dir = cfg.step_dir("test123", "02_separate")

        runner = PipelineRunner(cfg)
        with _pipeline_patches(sep_dir):
            state = runner.run(state)

        result = state.get_step(StepName.SEPARATE)
        assert result.started_at != ""
        assert result.finished_at != ""
        assert result.duration_sec >= 0.0
