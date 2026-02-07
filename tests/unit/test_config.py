"""Tests for yt_dbl.config — settings, paths, and RAM-based auto-detection."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from yt_dbl.config import (
    Settings,
    _detect_max_models,
    _detect_separation_batch_size,
    _get_total_ram_gb,
    _macos_ram_gb,
)


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

    def test_voice_ref_duration_validation(self) -> None:
        with pytest.raises(ValueError, match="greater than or equal to 3"):
            Settings(voice_ref_duration=1.0)
        with pytest.raises(ValueError, match="less than or equal to 30"):
            Settings(voice_ref_duration=60.0)

    def test_chunk_overlap_exceeds_chunk_raises(self) -> None:
        """overlap >= chunk causes infinite loop in chunked ASR — must reject."""
        with pytest.raises(ValueError, match="must be less than"):
            Settings(
                transcription_max_chunk_minutes=5.0,
                transcription_chunk_overlap_minutes=10.0,
            )

    def test_chunk_overlap_equals_chunk_raises(self) -> None:
        """overlap == chunk also produces zero step — must reject."""
        with pytest.raises(ValueError, match="must be less than"):
            Settings(
                transcription_max_chunk_minutes=5.0,
                transcription_chunk_overlap_minutes=5.0,
            )

    def test_chunk_overlap_less_than_chunk_ok(self) -> None:
        s = Settings(
            transcription_max_chunk_minutes=30.0,
            transcription_chunk_overlap_minutes=2.0,
        )
        assert s.transcription_chunk_overlap_minutes < s.transcription_max_chunk_minutes


# ── RAM-based auto-detection ────────────────────────────────────────────────


class TestDetectMaxModels:
    @pytest.mark.parametrize(
        ("ram_gb", "expected"),
        [
            (0.0, 1),
            (8.0, 1),
            (16.0, 1),
            (17.0, 2),
            (24.0, 2),
            (31.0, 2),
            (32.0, 3),
            (64.0, 3),
        ],
        ids=["zero", "8gb", "16gb", "17gb", "24gb", "31gb", "32gb", "64gb"],
    )
    @patch("yt_dbl.config._get_total_ram_gb")
    def test_ram_tiers(self, mock_ram: MagicMock, ram_gb: float, expected: int) -> None:
        mock_ram.return_value = ram_gb
        assert _detect_max_models() == expected


class TestDetectSeparationBatchSize:
    @pytest.mark.parametrize(
        ("ram_gb", "expected"),
        [
            (8.0, 1),
            (16.0, 1),
            (17.0, 2),
            (31.0, 2),
            (32.0, 4),
            (47.0, 4),
            (48.0, 8),
            (128.0, 8),
        ],
        ids=["8gb", "16gb", "17gb", "31gb", "32gb", "47gb", "48gb", "128gb"],
    )
    @patch("yt_dbl.config._get_total_ram_gb")
    def test_ram_tiers(self, mock_ram: MagicMock, ram_gb: float, expected: int) -> None:
        mock_ram.return_value = ram_gb
        assert _detect_separation_batch_size() == expected


class TestGetTotalRAM:
    @patch("os.sysconf")
    def test_via_sysconf(self, mock_sysconf: MagicMock) -> None:
        # 4M pages * 4096 bytes = 16 GiB
        mock_sysconf.side_effect = lambda name: {"SC_PAGE_SIZE": 4096, "SC_PHYS_PAGES": 4194304}[
            name
        ]
        assert _get_total_ram_gb() == pytest.approx(16.0)

    @patch("yt_dbl.config._macos_ram_gb", return_value=32.0)
    @patch("os.sysconf", side_effect=AttributeError)
    def test_fallback_to_macos(self, mock_sysconf: MagicMock, mock_macos: MagicMock) -> None:
        result = _get_total_ram_gb()
        assert result == 32.0
        mock_macos.assert_called_once()


class TestMacosRAM:
    @patch("subprocess.run")
    def test_parses_sysctl(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="17179869184\n",  # 16 GiB
        )
        assert _macos_ram_gb() == pytest.approx(16.0)

    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_returns_zero_on_failure(self, mock_run: MagicMock) -> None:
        assert _macos_ram_gb() == 0.0
        mock_run.assert_called_once()
