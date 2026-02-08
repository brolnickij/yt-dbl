"""Tests for yt_dbl.models.manager — LRU model management."""

import pytest

from yt_dbl.models.manager import ModelManager


class TestModelManager:
    def test_register_and_load(self) -> None:
        mgr = ModelManager(max_loaded=2)
        mgr.register("model_a", loader=lambda: "loaded_a")
        result = mgr.get("model_a")
        assert result == "loaded_a"
        assert "model_a" in mgr.loaded_names

    def test_lru_eviction(self) -> None:
        mgr = ModelManager(max_loaded=1)
        mgr.register("model_a", loader=lambda: "a")
        mgr.register("model_b", loader=lambda: "b")

        mgr.get("model_a")
        assert mgr.loaded_names == ["model_a"]

        mgr.get("model_b")
        assert mgr.loaded_names == ["model_b"]
        assert "model_a" not in mgr.loaded_names

    def test_lru_order(self) -> None:
        mgr = ModelManager(max_loaded=2)
        mgr.register("a", loader=lambda: "a")
        mgr.register("b", loader=lambda: "b")
        mgr.register("c", loader=lambda: "c")

        mgr.get("a")
        mgr.get("b")
        # Touch 'a' to make it recently used
        mgr.get("a")
        # Loading 'c' should evict 'b' (least recently used)
        mgr.get("c")
        assert "a" in mgr.loaded_names
        assert "c" in mgr.loaded_names
        assert "b" not in mgr.loaded_names

    def test_unload(self) -> None:
        unloaded: list[str] = []
        mgr = ModelManager(max_loaded=2)
        mgr.register("model_a", loader=lambda: "a", unloader=lambda m: unloaded.append(m))
        mgr.get("model_a")
        mgr.unload("model_a")
        assert "model_a" not in mgr.loaded_names
        assert unloaded == ["a"]

    def test_unload_all(self) -> None:
        mgr = ModelManager(max_loaded=3)
        for name in ["a", "b", "c"]:
            n = name
            mgr.register(n, loader=lambda n=n: n)  # type: ignore[misc]
            mgr.get(n)
        mgr.unload_all()
        assert mgr.loaded_names == []

    def test_unknown_model_raises(self) -> None:
        mgr = ModelManager(max_loaded=1)
        with pytest.raises(KeyError):
            mgr.get("nonexistent")

    def test_cached_hit_no_reload(self) -> None:
        call_count = 0

        def loader() -> str:
            nonlocal call_count
            call_count += 1
            return "model"

        mgr = ModelManager(max_loaded=2)
        mgr.register("m", loader=loader)
        mgr.get("m")
        mgr.get("m")
        mgr.get("m")
        assert call_count == 1

    def test_unload_nonexistent_is_safe(self) -> None:
        """Unloading a model that isn't loaded should not crash."""
        mgr = ModelManager(max_loaded=2)
        mgr.register("a", loader=lambda: "a")
        mgr.unload("a")  # not loaded yet — should be a no-op

    def test_eviction_calls_unloader(self) -> None:
        """When LRU evicts a model, the unloader callback fires."""
        unloaded: list[str] = []
        mgr = ModelManager(max_loaded=1)
        mgr.register("a", loader=lambda: "val_a", unloader=lambda m: unloaded.append(m))
        mgr.register("b", loader=lambda: "val_b")

        mgr.get("a")
        mgr.get("b")  # triggers eviction of 'a'
        assert "val_a" in unloaded

    def test_re_register_overwrites(self) -> None:
        """Re-registering a name should update the loader."""
        mgr = ModelManager(max_loaded=2)
        mgr.register("m", loader=lambda: "v1")
        assert mgr.get("m") == "v1"
        mgr.unload("m")
        mgr.register("m", loader=lambda: "v2")
        assert mgr.get("m") == "v2"

    def test_registered_names(self) -> None:
        """registered_names lists all registered (not just loaded) models."""
        mgr = ModelManager(max_loaded=2)
        mgr.register("a", loader=lambda: "a")
        mgr.register("b", loader=lambda: "b")
        assert set(mgr.registered_names) == {"a", "b"}
        assert mgr.loaded_names == []

    def test_cleanup_memory_mlx(self) -> None:
        """_cleanup_memory should not crash even without MLX/torch installed."""
        mgr = ModelManager(max_loaded=1)
        mgr._cleanup_memory()  # should not raise

    @pytest.mark.parametrize("bad_value", [0, -1, -100])
    def test_max_loaded_below_one_raises(self, bad_value: int) -> None:
        """ModelManager rejects max_loaded < 1 to prevent infinite eviction."""
        with pytest.raises(ValueError, match="max_loaded must be >= 1"):
            ModelManager(max_loaded=bad_value)
