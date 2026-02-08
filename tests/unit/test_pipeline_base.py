"""Tests for yt_dbl.pipeline.base — PipelineStep ABC and shared helpers."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from yt_dbl.pipeline.base import PipelineStep
from yt_dbl.schemas import PipelineState, StepName

# ── Concrete subclass for testing ───────────────────────────────────────────


class _DummyStep(PipelineStep):
    """Minimal concrete step for exercising base-class helpers."""

    name = StepName.DOWNLOAD
    description = "dummy"

    def run(self, state: PipelineState) -> PipelineState:
        return state


# ── _get_or_load_model tests ────────────────────────────────────────────────


class TestGetOrLoadModel:
    """Tests for the shared _get_or_load_model helper on PipelineStep."""

    def test_loads_via_manager_when_available(self, tmp_path: pytest.TempPathFactory) -> None:
        """When a ModelManager is present, the model is registered and loaded through it."""
        manager = MagicMock()
        manager.registered_names = []
        manager.get.return_value = "fake-model"

        step = _DummyStep(
            settings=MagicMock(),
            step_dir=tmp_path,  # type: ignore[arg-type]
            model_manager=manager,
        )
        loader = MagicMock(return_value="direct-model")

        result = step._get_or_load_model("my-model", loader=loader)

        manager.register.assert_called_once_with("my-model", loader=loader)
        manager.get.assert_called_once_with("my-model")
        loader.assert_not_called()  # loader is NOT called directly
        assert result == "fake-model"

    def test_skips_registration_when_already_registered(self, tmp_path: Any) -> None:
        """If the model is already registered, register() is not called again."""
        manager = MagicMock()
        manager.registered_names = ["my-model"]
        manager.get.return_value = "cached"

        step = _DummyStep(
            settings=MagicMock(),
            step_dir=tmp_path,
            model_manager=manager,
        )

        result = step._get_or_load_model("my-model", loader=MagicMock())

        manager.register.assert_not_called()
        manager.get.assert_called_once_with("my-model")
        assert result == "cached"

    def test_loads_directly_without_manager(self, tmp_path: Any) -> None:
        """Without a ModelManager, the loader is called directly."""
        step = _DummyStep(
            settings=MagicMock(),
            step_dir=tmp_path,
            model_manager=None,
        )
        loader = MagicMock(return_value="direct-model")

        result = step._get_or_load_model("my-model", loader=loader)

        loader.assert_called_once()
        assert result == "direct-model"

    def test_loader_exception_propagates(self, tmp_path: Any) -> None:
        """Errors from the loader bubble up unchanged."""
        step = _DummyStep(
            settings=MagicMock(),
            step_dir=tmp_path,
            model_manager=None,
        )
        loader = MagicMock(side_effect=RuntimeError("OOM"))

        with pytest.raises(RuntimeError, match="OOM"):
            step._get_or_load_model("broken", loader=loader)
