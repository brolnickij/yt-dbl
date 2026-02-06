"""Tests for yt_dbl.utils.memory — GPU memory cleanup."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from yt_dbl.utils.memory import cleanup_gpu_memory


class TestCleanupGpuMemory:
    def test_calls_gc_collect(self) -> None:
        with patch("yt_dbl.utils.memory.gc.collect") as mock_gc:
            cleanup_gpu_memory()
        mock_gc.assert_called_once()

    def test_survives_no_gpu_libraries(self) -> None:
        """Should not crash if neither mlx nor torch are installed."""
        cleanup_gpu_memory()  # must not raise

    def test_clears_mlx_metal_cache(self) -> None:
        try:
            import mlx.core as mx  # noqa: F401
        except ImportError:
            pytest.skip("mlx not installed")
        with patch("mlx.core.metal.clear_cache") as mock_clear:
            cleanup_gpu_memory()
        mock_clear.assert_called_once()

    def test_clears_torch_mps_cache(self) -> None:
        mock_torch = MagicMock()
        mock_torch.backends.mps.is_available.return_value = True
        mock_torch.cuda.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": mock_torch}):
            cleanup_gpu_memory()
        mock_torch.mps.empty_cache.assert_called_once()

    def test_clears_torch_cuda_cache(self) -> None:
        mock_torch = MagicMock()
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = True
        with patch.dict("sys.modules", {"torch": mock_torch}):
            cleanup_gpu_memory()
        mock_torch.cuda.empty_cache.assert_called_once()

    def test_mps_preferred_over_cuda(self) -> None:
        """When both MPS and CUDA are available, MPS takes priority."""
        mock_torch = MagicMock()
        mock_torch.backends.mps.is_available.return_value = True
        mock_torch.cuda.is_available.return_value = True
        with patch.dict("sys.modules", {"torch": mock_torch}):
            cleanup_gpu_memory()
        mock_torch.mps.empty_cache.assert_called_once()
        mock_torch.cuda.empty_cache.assert_not_called()
