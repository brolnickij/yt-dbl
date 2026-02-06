"""Tests for yt_dbl.config — settings and paths."""

from pathlib import Path

import pytest

from yt_dbl.config import Settings


class TestSettings:
    def test_defaults(self) -> None:
        s = Settings(anthropic_api_key="test", hf_token="test")
        assert s.target_language == "ru"
        assert s.output_format == "mp4"
        assert s.subtitle_mode == "softsub"
        assert s.background_volume == 0.15
        assert s.max_speed_factor == 1.4
        assert s.voice_ref_duration == 7.0
        assert s.max_loaded_models >= 1  # auto-detected based on RAM
        assert s.separation_batch_size >= 1  # auto-detected based on RAM
        assert s.sample_rate == 48000
        assert s.claude_model == "claude-sonnet-4-5"

    def test_max_loaded_models_auto(self) -> None:
        """Default 0 triggers auto-detection → always ≥ 1."""
        s = Settings()
        assert s.max_loaded_models >= 1

    def test_max_loaded_models_explicit(self) -> None:
        """Explicit value overrides auto-detection."""
        s = Settings(max_loaded_models=5)
        assert s.max_loaded_models == 5

    def test_separation_batch_size_auto(self) -> None:
        """Default 0 triggers auto-detection → always ≥ 1."""
        s = Settings()
        assert s.separation_batch_size >= 1

    def test_separation_batch_size_explicit(self) -> None:
        """Explicit value overrides auto-detection."""
        s = Settings(separation_batch_size=3)
        assert s.separation_batch_size == 3

    def test_job_dir(self, tmp_path: Path) -> None:
        s = Settings(work_dir=tmp_path / "work")
        d = s.job_dir("abc123")
        assert d.exists()
        assert d.name == "abc123"

    def test_step_dir(self, tmp_path: Path) -> None:
        s = Settings(work_dir=tmp_path / "work")
        d = s.step_dir("abc123", "01_download")
        assert d.exists()
        assert d.name == "01_download"
        assert d.parent.name == "abc123"

    def test_background_volume_validation(self) -> None:
        with pytest.raises(ValueError, match="less than or equal to 1"):
            Settings(background_volume=1.5)

    def test_max_speed_validation(self) -> None:
        with pytest.raises(ValueError, match="greater than or equal to 1"):
            Settings(max_speed_factor=0.5)
