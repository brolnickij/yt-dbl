"""Model manager with LRU eviction for ML models.

On machines with limited RAM (16 GB), only one large model should be loaded
at a time.  On 48 GB+ machines, several models can coexist.  The manager
tracks loaded models and evicts the least-recently-used one when the limit
is reached.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from yt_dbl.utils.logging import get_metal_memory_mb, log_info, log_model_load, log_model_unload

__all__ = ["ModelManager"]


@dataclass
class LoadedModel:
    """Metadata for a model currently held in memory."""

    name: str
    model: Any
    loaded_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.last_used = time.time()


class ModelManager:
    """LRU cache for ML models with configurable capacity.

    Usage:
        manager = ModelManager(max_loaded=2)

        # Register a loader
        manager.register("qwen3-asr", loader_fn, unloader_fn)

        # Get a model (loads if needed, evicts LRU if over limit)
        model = manager.get("qwen3-asr")

        # Explicitly unload
        manager.unload("qwen3-asr")

        # Unload everything
        manager.unload_all()
    """

    def __init__(self, max_loaded: int = 2) -> None:
        self.max_loaded = max_loaded
        self._models: OrderedDict[str, LoadedModel] = OrderedDict()
        self._loaders: dict[str, tuple[Any, Any]] = {}  # name → (load_fn, unload_fn)

    def register(
        self,
        name: str,
        loader: Any,  # Callable[[], Any]
        unloader: Any | None = None,  # Callable[[Any], None]
    ) -> None:
        """Register a model with its loader and optional unloader."""
        self._loaders[name] = (loader, unloader)

    def get(self, name: str) -> Any:
        """Get a loaded model, loading it if necessary."""
        if name in self._models:
            entry = self._models[name]
            entry.touch()
            self._models.move_to_end(name)
            return entry.model

        return self._load(name)

    def _load(self, name: str) -> Any:
        """Load a model, evicting LRU if over limit."""
        while len(self._models) >= self.max_loaded:
            self._evict_lru()

        if name not in self._loaders:
            raise KeyError(
                f"Model '{name}' is not registered. Available: {list(self._loaders.keys())}"
            )

        loader, _ = self._loaders[name]

        mem_before = get_metal_memory_mb()
        t0 = time.time()

        model = loader()

        elapsed = time.time() - t0
        mem_after = get_metal_memory_mb()
        mem_delta = max(0.0, mem_after - mem_before)

        self._models[name] = LoadedModel(name=name, model=model)
        self._models.move_to_end(name)

        log_model_load(name, elapsed=elapsed, mem_delta_mb=mem_delta)
        log_info(f"Models loaded: {len(self._models)}/{self.max_loaded}")
        return model

    def _evict_lru(self) -> None:
        """Evict the least recently used model."""
        if not self._models:
            return

        # First item = least recently used
        name, _entry = next(iter(self._models.items()))
        self.unload(name)

    def unload(self, name: str) -> None:
        """Explicitly unload a model and free memory."""
        if name not in self._models:
            return

        entry = self._models.pop(name)

        mem_before = get_metal_memory_mb()

        _, unloader = self._loaders.get(name, (None, None))
        if unloader:
            unloader(entry.model)

        del entry.model
        self._cleanup_memory()

        mem_after = get_metal_memory_mb()
        mem_freed = max(0.0, mem_before - mem_after)
        log_model_unload(name, mem_freed_mb=mem_freed)

    def unload_all(self) -> None:
        """Unload all models."""
        for name in list(self._models.keys()):
            self.unload(name)

    @property
    def loaded_names(self) -> list[str]:
        return list(self._models.keys())

    @property
    def registered_names(self) -> list[str]:
        return list(self._loaders.keys())

    @staticmethod
    def _cleanup_memory() -> None:
        """Run garbage collection and clear GPU/Metal caches."""
        from yt_dbl.utils.memory import cleanup_gpu_memory

        cleanup_gpu_memory()
