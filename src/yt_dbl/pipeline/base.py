"""Abstract base class for pipeline steps."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from yt_dbl.config import Settings
    from yt_dbl.models.manager import ModelManager
    from yt_dbl.schemas import PipelineState, StepName


class PipelineStep(ABC):
    """Base class every pipeline step must implement.

    Each step:
      - receives the current PipelineState
      - has access to its own working directory
      - can use the shared ModelManager for ML model lifecycle
      - writes outputs to that directory
      - updates PipelineState with results
    """

    # Subclasses must set these
    name: StepName
    description: str = ""

    def __init__(
        self,
        settings: Settings,
        work_dir: Path,
        model_manager: ModelManager | None = None,
    ) -> None:
        self.settings = settings
        self.work_dir = work_dir
        self.model_manager = model_manager

    @abstractmethod
    def run(self, state: PipelineState) -> PipelineState:
        """Execute the step and return the updated state.

        Implementations should:
          1. Read inputs from state / previous step directories
          2. Process
          3. Write outputs to self.work_dir
          4. Update state with results (segments, speakers, etc.)
          5. Return state
        """
        ...

    def validate_inputs(self, state: PipelineState) -> None:  # noqa: B027
        """Optional: validate that required inputs exist before running.

        Raise ValueError with a clear message if something is missing.
        """

    @property
    def step_dir(self) -> Path:
        """Convenience alias for the step working directory."""
        return self.work_dir
