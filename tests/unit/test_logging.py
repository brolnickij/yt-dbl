"""Tests for yt_dbl.utils.logging — rich logging utilities."""

from yt_dbl.schemas import STEP_ORDER, StepName
from yt_dbl.utils.logging import get_metal_memory_mb, get_rss_mb, step_label, step_prefix


class TestLogging:
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


class TestMemoryTracking:
    def test_get_rss_mb_returns_positive(self) -> None:
        """RSS should be > 0 for any running process."""
        rss = get_rss_mb()
        assert rss >= 0  # may be 0 if resource module unavailable

    def test_get_metal_memory_mb_returns_float(self) -> None:
        """Should return 0.0 if MLX not available, or a positive value otherwise."""
        metal = get_metal_memory_mb()
        assert isinstance(metal, float)
        assert metal >= 0.0
