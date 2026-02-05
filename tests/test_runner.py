"""Tests for pipeline runner — checkpointing and resume."""

import json
from pathlib import Path

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
    def test_full_stub_pipeline(self, tmp_path: Path) -> None:
        cfg = Settings(work_dir=tmp_path / "work")
        state = PipelineState(
            video_id="test123",
            url="https://youtube.com/watch?v=test123",
            target_language="ru",
        )

        runner = PipelineRunner(cfg)
        state = runner.run(state)

        assert state.next_step is None
        for step_name in [
            StepName.DOWNLOAD,
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

        # Complete first 3 steps
        runner = PipelineRunner(cfg)
        state = runner.run(state)

        # Mark last 3 as pending again
        for name in [StepName.TRANSLATE, StepName.SYNTHESIZE, StepName.ASSEMBLE]:
            state.get_step(name).status = StepStatus.PENDING
        save_state(state, cfg)

        # Resume should skip the first 3
        loaded = load_state(cfg, "test123")
        assert loaded is not None
        assert loaded.next_step == StepName.TRANSLATE

    def test_from_step(self, tmp_path: Path) -> None:
        cfg = Settings(work_dir=tmp_path / "work")
        state = PipelineState(video_id="test123", url="https://example.com")

        # Run full pipeline first
        runner = PipelineRunner(cfg)
        state = runner.run(state)

        # Re-run from translate
        state = runner.run(state, from_step=StepName.TRANSLATE)
        assert state.get_step(StepName.TRANSLATE).status == StepStatus.COMPLETED
