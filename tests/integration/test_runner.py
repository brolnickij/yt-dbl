"""Tests for pipeline runner — checkpointing and resume."""

import json
from pathlib import Path
from unittest.mock import patch

from tests.conftest import prefill_download
from yt_dbl.config import Settings
from yt_dbl.pipeline.runner import PipelineRunner, load_state, save_state
from yt_dbl.schemas import PipelineState, StepName, StepStatus


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
        cfg = Settings(work_dir=tmp_path / "work")
        state = PipelineState(
            video_id="test123",
            url="https://youtube.com/watch?v=test123",
            target_language="ru",
        )
        state = prefill_download(state, cfg)
        save_state(state, cfg)

        runner = PipelineRunner(cfg)
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
        cfg = Settings(work_dir=tmp_path / "work")
        state = PipelineState(video_id="test123", url="https://example.com")
        state = prefill_download(state, cfg)

        runner = PipelineRunner(cfg)
        state = runner.run(state)

        # Mark last 3 as pending again
        for name in [StepName.TRANSLATE, StepName.SYNTHESIZE, StepName.ASSEMBLE]:
            state.get_step(name).status = StepStatus.PENDING
        save_state(state, cfg)

        loaded = load_state(cfg, "test123")
        assert loaded is not None
        assert loaded.next_step == StepName.TRANSLATE

    def test_from_step(self, tmp_path: Path) -> None:
        cfg = Settings(work_dir=tmp_path / "work")
        state = PipelineState(video_id="test123", url="https://example.com")
        state = prefill_download(state, cfg)

        runner = PipelineRunner(cfg)
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
        cfg = Settings(work_dir=tmp_path / "work")
        state = PipelineState(video_id="test123", url="https://example.com")
        state = prefill_download(state, cfg)

        runner = PipelineRunner(cfg)
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
        cfg = Settings(work_dir=tmp_path / "work")
        state = PipelineState(video_id="test123", url="https://example.com")
        state = prefill_download(state, cfg)

        runner = PipelineRunner(cfg)
        state = runner.run(state)

        result = state.get_step(StepName.SEPARATE)
        assert result.started_at != ""
        assert result.finished_at != ""
        assert result.duration_sec >= 0.0
