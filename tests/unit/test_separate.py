"""Tests for yt_dbl.pipeline.separate — vocal separation step (mocked)."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from yt_dbl.config import Settings
from yt_dbl.pipeline.base import SeparationError, StepValidationError
from yt_dbl.pipeline.separate import SeparateStep
from yt_dbl.schemas import STEP_DIRS, PipelineState, StepName, StepStatus

if TYPE_CHECKING:
    from pathlib import Path


class TestSeparateStepValidation:
    def _make(self, tmp_path: Path) -> tuple[SeparateStep, Settings, PipelineState]:
        cfg = Settings(work_dir=tmp_path / "work")
        step_dir = cfg.step_dir("test123", STEP_DIRS[StepName.SEPARATE])
        step = SeparateStep(settings=cfg, work_dir=step_dir)
        state = PipelineState(video_id="test123", url="https://example.com")
        return step, cfg, state

    def test_validate_missing_audio_output(self, tmp_path: Path) -> None:
        step, _, state = self._make(tmp_path)
        # download step has no outputs
        with pytest.raises(StepValidationError, match="No audio file"):
            step.validate_inputs(state)

    def test_validate_audio_file_not_found(self, tmp_path: Path) -> None:
        step, _, state = self._make(tmp_path)
        dl = state.get_step(StepName.DOWNLOAD)
        dl.status = StepStatus.COMPLETED
        dl.outputs = {"video": "video.mp4", "audio": "audio.wav"}
        # File doesn't actually exist on disk
        with pytest.raises(StepValidationError, match="Audio file not found"):
            step.validate_inputs(state)

    def test_validate_ok(self, tmp_path: Path) -> None:
        step, cfg, state = self._make(tmp_path)
        dl = state.get_step(StepName.DOWNLOAD)
        dl.status = StepStatus.COMPLETED
        dl.outputs = {"video": "video.mp4", "audio": "audio.wav"}
        # Create the actual file
        dl_dir = cfg.step_dir("test123", STEP_DIRS[StepName.DOWNLOAD])
        (dl_dir / "audio.wav").write_bytes(b"fake-audio")
        step.validate_inputs(state)  # Should not raise


class TestSeparateStepRun:
    def _setup(self, tmp_path: Path) -> tuple[SeparateStep, Settings, PipelineState]:
        cfg = Settings(work_dir=tmp_path / "work")
        step_dir = cfg.step_dir("test123", STEP_DIRS[StepName.SEPARATE])
        step = SeparateStep(settings=cfg, work_dir=step_dir)
        state = PipelineState(video_id="test123", url="https://example.com")
        # Prefill download
        dl = state.get_step(StepName.DOWNLOAD)
        dl.status = StepStatus.COMPLETED
        dl.outputs = {"video": "video.mp4", "audio": "audio.wav"}
        dl_dir = cfg.step_dir("test123", STEP_DIRS[StepName.DOWNLOAD])
        (dl_dir / "audio.wav").write_bytes(b"fake-audio-content")
        return step, cfg, state

    @patch("yt_dbl.pipeline.separate.SeparateStep._run_separation")
    def test_run_success(
        self,
        mock_sep: MagicMock,
        tmp_path: Path,
    ) -> None:
        step, _cfg, state = self._setup(tmp_path)

        # Fake separation: create output files
        def fake_separation(audio_path: Path) -> None:
            (step.step_dir / "vocals.wav").write_bytes(b"separated-vocals")
            (step.step_dir / "background.wav").write_bytes(b"separated-bg")

        mock_sep.side_effect = fake_separation

        state = step.run(state)

        result = state.get_step(StepName.SEPARATE)
        assert result.outputs["vocals"] == "vocals.wav"
        assert result.outputs["background"] == "background.wav"
        mock_sep.assert_called_once()

    @patch("yt_dbl.pipeline.separate.SeparateStep._run_separation")
    def test_run_idempotent_skips_existing(
        self,
        mock_sep: MagicMock,
        tmp_path: Path,
    ) -> None:
        step, _cfg, state = self._setup(tmp_path)

        # Pre-create output files
        (step.step_dir / "vocals.wav").write_bytes(b"existing-vocals")
        (step.step_dir / "background.wav").write_bytes(b"existing-bg")

        state = step.run(state)

        # Should NOT call separation
        mock_sep.assert_not_called()
        assert state.get_step(StepName.SEPARATE).outputs["vocals"] == "vocals.wav"

    @patch("yt_dbl.pipeline.separate.SeparateStep._run_separation")
    def test_run_raises_if_vocals_missing(
        self,
        mock_sep: MagicMock,
        tmp_path: Path,
    ) -> None:
        step, _cfg, state = self._setup(tmp_path)

        # Separation runs but doesn't create vocals
        def fake_no_output(audio_path: Path) -> None:
            pass

        mock_sep.side_effect = fake_no_output

        with pytest.raises(SeparationError, match="Vocals file was not created"):
            step.run(state)

    @patch("yt_dbl.pipeline.separate.SeparateStep._run_separation")
    def test_run_raises_if_background_missing(
        self,
        mock_sep: MagicMock,
        tmp_path: Path,
    ) -> None:
        step, _cfg, state = self._setup(tmp_path)

        # Only creates vocals, not background
        def fake_partial(audio_path: Path) -> None:
            (step.step_dir / "vocals.wav").write_bytes(b"vocals")

        mock_sep.side_effect = fake_partial

        with pytest.raises(SeparationError, match="Background file was not created"):
            step.run(state)


class TestRenameOutputs:
    def _make_step(self, tmp_path: Path) -> SeparateStep:
        cfg = Settings(work_dir=tmp_path / "work")
        step_dir = cfg.step_dir("test123", STEP_DIRS[StepName.SEPARATE])
        return SeparateStep(settings=cfg, work_dir=step_dir)

    def test_renames_standard_output_files(self, tmp_path: Path) -> None:
        step = self._make_step(tmp_path)
        # Create files with audio-separator naming convention
        vocals_orig = step.step_dir / "audio_(Vocals)_model_bs_roformer.wav"
        instr_orig = step.step_dir / "audio_(Instrumental)_model_bs_roformer.wav"
        vocals_orig.write_bytes(b"vocals-data")
        instr_orig.write_bytes(b"instr-data")

        step._rename_outputs([str(vocals_orig), str(instr_orig)])

        assert (step.step_dir / "vocals.wav").exists()
        assert (step.step_dir / "background.wav").exists()
        assert not vocals_orig.exists()  # original removed
        assert not instr_orig.exists()

    def test_raises_if_no_vocals_in_output(self, tmp_path: Path) -> None:
        step = self._make_step(tmp_path)
        instr = step.step_dir / "audio_(Instrumental)_model.wav"
        instr.write_bytes(b"data")

        with pytest.raises(SeparationError, match="No vocals stem"):
            step._rename_outputs([str(instr)])

    def test_raises_if_no_instrumental_in_output(self, tmp_path: Path) -> None:
        step = self._make_step(tmp_path)
        vocals = step.step_dir / "audio_(Vocals)_model.wav"
        vocals.write_bytes(b"data")

        with pytest.raises(SeparationError, match="No instrumental stem"):
            step._rename_outputs([str(vocals)])

    def test_handles_case_insensitive_labels(self, tmp_path: Path) -> None:
        step = self._make_step(tmp_path)
        vocals = step.step_dir / "audio_(VOCALS)_model.wav"
        instr = step.step_dir / "audio_(INSTRUMENTAL)_model.wav"
        vocals.write_bytes(b"v")
        instr.write_bytes(b"i")

        step._rename_outputs([str(vocals), str(instr)])

        assert (step.step_dir / "vocals.wav").exists()
        assert (step.step_dir / "background.wav").exists()

    def test_warns_on_missing_file(self, tmp_path: Path) -> None:
        step = self._make_step(tmp_path)
        # One file exists, one doesn't
        vocals = step.step_dir / "audio_(Vocals)_model.wav"
        vocals.write_bytes(b"v")
        nonexistent = str(step.step_dir / "audio_(Instrumental)_ghost.wav")

        with pytest.raises(SeparationError, match="No instrumental stem"):
            step._rename_outputs([str(vocals), nonexistent])

    def test_handles_relative_paths(self, tmp_path: Path) -> None:
        """audio-separator may return just filenames — resolve against step_dir."""
        step = self._make_step(tmp_path)
        vocals = step.step_dir / "audio_(Vocals)_model.wav"
        instr = step.step_dir / "audio_(Instrumental)_model.wav"
        vocals.write_bytes(b"v")
        instr.write_bytes(b"i")

        # Pass only filenames (relative), not absolute paths
        step._rename_outputs(["audio_(Vocals)_model.wav", "audio_(Instrumental)_model.wav"])

        assert (step.step_dir / "vocals.wav").exists()
        assert (step.step_dir / "background.wav").exists()


class TestSeparationConfig:
    def test_default_model(self) -> None:
        cfg = Settings()
        assert "roformer" in cfg.separation_model.lower()

    def test_default_segment_size(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("YT_DBL_SEPARATION_SEGMENT_SIZE", raising=False)
        cfg = Settings(_env_file=None)  # type: ignore[call-arg]
        assert cfg.separation_segment_size == 256

    def test_default_overlap(self) -> None:
        cfg = Settings()
        assert cfg.separation_overlap == 8

    def test_model_cache_dir(self) -> None:
        cfg = Settings()
        assert "yt-dbl" in str(cfg.model_cache_dir)

    def test_custom_model_via_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("YT_DBL_SEPARATION_MODEL", "custom_model.ckpt")
        cfg = Settings()
        assert cfg.separation_model == "custom_model.ckpt"
