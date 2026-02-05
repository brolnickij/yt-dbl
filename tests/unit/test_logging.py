"""Tests for yt_dbl.utils.logging — rich logging utilities."""

from yt_dbl.schemas import STEP_ORDER, StepName
from yt_dbl.utils.logging import step_label, step_prefix


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
