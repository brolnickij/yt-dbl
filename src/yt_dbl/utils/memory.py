"""GPU / accelerator memory management utilities.

Centralises cache-clearing for MLX Metal, PyTorch MPS and CUDA so that
every callsite (``ModelManager``, ``SeparateStep``, …) uses the same
logic without duplication.
"""

from __future__ import annotations

import gc

__all__ = ["cleanup_gpu_memory"]


def cleanup_gpu_memory() -> None:
    """Run garbage collection and clear GPU / accelerator caches.

    Handles three backends, importing each lazily so the function works
    regardless of which libraries are installed:

    * **MLX Metal** — Apple Silicon Neural Engine / GPU cache
    * **PyTorch MPS** — Apple Metal Performance Shaders
    * **PyTorch CUDA** — NVIDIA GPU
    """
    gc.collect()

    # MLX Metal cache (Apple Silicon)
    try:
        import mlx.core as mx

        if hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
    except ImportError:
        pass

    # PyTorch MPS / CUDA cache (audio-separator uses torch)
    try:
        import torch

        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
