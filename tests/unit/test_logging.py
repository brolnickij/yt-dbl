"""Tests for yt_dbl.utils.logging — rich logging utilities."""

from __future__ import annotations

import logging
import os
import time
from unittest.mock import MagicMock, patch

import pytest
from rich.progress import Progress

from yt_dbl.schemas import STEP_ORDER, StepName
from yt_dbl.utils.logging import (
    create_progress,
    get_metal_memory_mb,
    get_rss_mb,
    log_info,
    log_memory_status,
    log_model_load,
    log_model_unload,
    log_step_done,
    log_step_fail,
    log_step_skip,
    log_step_start,
    log_warning,
    step_label,
    step_prefix,
    suppress_library_noise,
    timer,
)

# ── Step formatting ─────────────────────────────────────────────────────────


class TestStepFormatting:
    def test_step_prefix_format(self) -> None:
        prefix = step_prefix(StepName.DOWNLOAD)
        assert "1/6" in prefix

    def test_step_prefix_last(self) -> None:
        prefix = step_prefix(StepName.ASSEMBLE)
        assert "6/6" in prefix

    def test_step_labels_all_defined(self) -> None:
        for step in STEP_ORDER:
            label = step_label(step)
            assert isinstance(label, str)
            assert len(label) > 0


# ── Log functions ───────────────────────────────────────────────────────────


class TestLogFunctions:
    @patch("yt_dbl.utils.logging.console")
    def test_log_step_start_with_detail(self, mock_console: MagicMock) -> None:
        log_step_start(StepName.DOWNLOAD, detail="video.mp4")
        mock_console.print.assert_called_once()
        msg: str = mock_console.print.call_args[0][0]
        assert "Download" in msg
        assert "video.mp4" in msg

    @patch("yt_dbl.utils.logging.console")
    def test_log_step_start_no_detail(self, mock_console: MagicMock) -> None:
        log_step_start(StepName.DOWNLOAD)
        msg: str = mock_console.print.call_args[0][0]
        assert "Download" in msg
        assert "—" not in msg

    @patch("yt_dbl.utils.logging.console")
    def test_log_step_done(self, mock_console: MagicMock) -> None:
        log_step_done(StepName.SEPARATE, elapsed=5.3)
        msg: str = mock_console.print.call_args[0][0]
        assert "Separate" in msg
        assert "5.3s" in msg
        assert "✓" in msg

    @patch("yt_dbl.utils.logging.console")
    def test_log_step_skip(self, mock_console: MagicMock) -> None:
        log_step_skip(StepName.TRANSCRIBE)
        msg: str = mock_console.print.call_args[0][0]
        assert "Transcribe" in msg
        assert "skipped" in msg

    @patch("yt_dbl.utils.logging.console")
    def test_log_step_fail(self, mock_console: MagicMock) -> None:
        log_step_fail(StepName.TRANSLATE, error="API timeout")
        msg: str = mock_console.print.call_args[0][0]
        assert "Translate" in msg
        assert "API timeout" in msg
        assert "✗" in msg

    @patch("yt_dbl.utils.logging.console")
    def test_log_info(self, mock_console: MagicMock) -> None:
        log_info("Processing 42 segments")
        msg: str = mock_console.print.call_args[0][0]
        assert "Processing 42 segments" in msg

    @patch("yt_dbl.utils.logging.console")
    def test_log_warning(self, mock_console: MagicMock) -> None:
        log_warning("Low memory")
        msg: str = mock_console.print.call_args[0][0]
        assert "Low memory" in msg
        assert "⚠" in msg

    @patch("yt_dbl.utils.logging.console")
    def test_log_model_load(self, mock_console: MagicMock) -> None:
        log_model_load("VibeVoice", elapsed=2.5, mem_delta_mb=1500)
        msg: str = mock_console.print.call_args[0][0]
        assert "VibeVoice" in msg
        assert "2.5s" in msg
        assert "1500" in msg

    @patch("yt_dbl.utils.logging.console")
    def test_log_model_unload(self, mock_console: MagicMock) -> None:
        log_model_unload("VibeVoice", mem_freed_mb=1500)
        msg: str = mock_console.print.call_args[0][0]
        assert "VibeVoice" in msg
        assert "1500" in msg

    @patch("yt_dbl.utils.logging.console")
    def test_log_memory_status(self, mock_console: MagicMock) -> None:
        log_memory_status()
        mock_console.print.assert_called_once()
        msg: str = mock_console.print.call_args[0][0]
        assert "Memory" in msg
        assert "RSS=" in msg


# ── Memory tracking ─────────────────────────────────────────────────────────


class TestMemoryTracking:
    def test_get_rss_mb_positive_for_running_process(self) -> None:
        """RSS must be > 0 for any running Python process on a supported platform."""
        rss = get_rss_mb()
        assert rss > 0

    def test_get_metal_memory_mb_returns_float(self) -> None:
        """Should return 0.0 if MLX not available, or a positive value otherwise."""
        metal = get_metal_memory_mb()
        assert isinstance(metal, float)
        assert metal >= 0.0


# ── Timer ───────────────────────────────────────────────────────────────────


class TestTimer:
    def test_tracks_elapsed_time(self) -> None:
        with timer() as t:
            time.sleep(0.05)
        assert t["elapsed"] >= 0.04

    def test_initial_elapsed_is_zero(self) -> None:
        with timer() as t:
            assert t["elapsed"] == 0.0


# ── Progress bar ────────────────────────────────────────────────────────────


class TestCreateProgress:
    def test_returns_progress_instance(self) -> None:
        p = create_progress()
        assert isinstance(p, Progress)


# ── Library noise suppression ───────────────────────────────────────────────


class TestSuppressLibraryNoise:
    def test_sets_and_restores_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Env vars are overridden inside the context and restored on exit."""
        monkeypatch.delenv("HF_HUB_DISABLE_PROGRESS_BARS", raising=False)
        monkeypatch.delenv("TRANSFORMERS_NO_ADVISORY_WARNINGS", raising=False)
        monkeypatch.delenv("TOKENIZERS_PARALLELISM", raising=False)

        with suppress_library_noise():
            assert os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"
            assert os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] == "1"
            assert os.environ["TOKENIZERS_PARALLELISM"] == "false"

        # Restored: keys should be absent since they were unset before
        assert "HF_HUB_DISABLE_PROGRESS_BARS" not in os.environ

    def test_silences_and_restores_loggers(self) -> None:
        """Noisy loggers are set to ERROR inside context and restored on exit."""
        logger = logging.getLogger("huggingface_hub")
        original_level = logger.level

        with suppress_library_noise():
            assert logger.level == logging.ERROR

        assert logger.level == original_level

    def test_restores_on_exception(self) -> None:
        """Cleanup happens even when an exception occurs inside the context."""
        logger = logging.getLogger("huggingface_hub")
        original_level = logger.level

        with pytest.raises(RuntimeError, match="boom"), suppress_library_noise():
            raise RuntimeError("boom")

        assert logger.level == original_level
