"""Tests for yt_dbl.utils.memory — GPU memory cleanup.

Every test fully mocks both ``mlx`` and ``torch`` to avoid real GPU / Metal
initialisation.  Importing the real ``mlx.core`` C-extension triggers Metal
GPU init and can abort (``Fatal Python error: Aborted``) when multiple xdist
workers race on GPU access simultaneously.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from yt_dbl.utils.memory import cleanup_gpu_memory


def _make_mock_mlx() -> MagicMock:
    """Return a ``MagicMock`` that behaves like ``mlx.core``."""
    mock_mx = MagicMock()
    mock_mx.metal.clear_cache = MagicMock()
    return mock_mx


def _make_mock_torch(*, mps_available: bool = False, cuda_available: bool = False) -> MagicMock:
    """Return a ``MagicMock`` that behaves like ``torch``."""
    mock_torch = MagicMock()
    mock_torch.backends.mps.is_available.return_value = mps_available
    mock_torch.cuda.is_available.return_value = cuda_available
    return mock_torch


class TestCleanupGpuMemory:
    def test_calls_gc_collect(self) -> None:
        with (
            patch.dict("sys.modules", {"mlx": None, "mlx.core": None, "torch": None}),
            patch("yt_dbl.utils.memory.gc.collect") as mock_gc,
        ):
            cleanup_gpu_memory()
        mock_gc.assert_called_once()

    def test_survives_no_gpu_libraries(self) -> None:
        """Should not crash if neither mlx nor torch are installed."""
        with patch.dict("sys.modules", {"mlx": None, "mlx.core": None, "torch": None}):
            cleanup_gpu_memory()  # must not raise

    def test_clears_mlx_metal_cache(self) -> None:
        mock_mx = _make_mock_mlx()
        mock_mlx_pkg = MagicMock()
        mock_mlx_pkg.core = mock_mx  # import mlx.core → mlx = __import__; mlx.core
        with patch.dict(
            "sys.modules",
            {"mlx": mock_mlx_pkg, "mlx.core": mock_mx, "torch": None},
        ):
            cleanup_gpu_memory()
        mock_mx.metal.clear_cache.assert_called_once()

    def test_clears_torch_mps_cache(self) -> None:
        mock_torch = _make_mock_torch(mps_available=True)
        with patch.dict(
            "sys.modules",
            {"mlx": None, "mlx.core": None, "torch": mock_torch},
        ):
            cleanup_gpu_memory()
        mock_torch.mps.empty_cache.assert_called_once()

    def test_clears_torch_cuda_cache(self) -> None:
        mock_torch = _make_mock_torch(cuda_available=True)
        with patch.dict(
            "sys.modules",
            {"mlx": None, "mlx.core": None, "torch": mock_torch},
        ):
            cleanup_gpu_memory()
        mock_torch.cuda.empty_cache.assert_called_once()

    def test_mps_preferred_over_cuda(self) -> None:
        """When both MPS and CUDA are available, MPS takes priority."""
        mock_torch = _make_mock_torch(mps_available=True, cuda_available=True)
        with patch.dict(
            "sys.modules",
            {"mlx": None, "mlx.core": None, "torch": mock_torch},
        ):
            cleanup_gpu_memory()
        mock_torch.mps.empty_cache.assert_called_once()
        mock_torch.cuda.empty_cache.assert_not_called()
