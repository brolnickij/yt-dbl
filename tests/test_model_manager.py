"""Tests for yt_dbl.models.manager — LRU model management."""

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
            mgr.register(n, loader=lambda n=n: n)
            mgr.get(n)
        mgr.unload_all()
        assert mgr.loaded_names == []

    def test_unknown_model_raises(self) -> None:
        mgr = ModelManager(max_loaded=1)
        try:
            mgr.get("nonexistent")
            raise AssertionError("Should have raised KeyError")
        except KeyError:
            pass

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
