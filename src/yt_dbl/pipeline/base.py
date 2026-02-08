"""Abstract base class for pipeline steps and exception hierarchy."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from yt_dbl.config import Settings
    from yt_dbl.models.manager import ModelManager
    from yt_dbl.schemas import PipelineState

from yt_dbl.schemas import STEP_DIRS, StepName

_T = TypeVar("_T")

# ── Exception hierarchy ─────────────────────────────────────────────────────


class PipelineStepError(Exception):
    """Base exception for all pipeline step errors."""


class StepValidationError(PipelineStepError):
    """Raised when a step's ``validate_inputs`` check fails."""


class DownloadError(PipelineStepError):
    """Raised when video or audio download fails."""


class SeparationError(PipelineStepError):
    """Raised when audio separation fails."""


class TranscriptionError(PipelineStepError):
    """Raised when ASR transcription or forced alignment fails."""


class TranslationError(PipelineStepError):
    """Raised when API translation fails or returns invalid data."""


class SynthesisError(PipelineStepError):
    """Raised when TTS voice synthesis fails."""


class AssemblyError(PipelineStepError):
    """Raised when final video assembly fails."""


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

        Raise ``StepValidationError`` with a clear message if something is missing.
        """

    def resolve_step_file(
        self,
        state: PipelineState,
        step: StepName,
        output_key: str,
    ) -> Path:
        """Resolve a file path from a previous step's outputs.

        Uses ``STEP_DIRS`` to build the correct directory without
        hardcoding step directory names across the codebase.
        """
        step_result = state.get_step(step)
        step_dir = self.settings.step_dir(state.video_id, STEP_DIRS[step])
        return step_dir / step_result.outputs[output_key]

    def _get_or_load_model(
        self,
        name: str,
        loader: Callable[[], _T],
    ) -> _T:
        """Load a model via *ModelManager* (with LRU eviction) or directly.

        When a ``model_manager`` is available the model is registered
        (if not already) and retrieved through the manager so that LRU
        eviction and memory tracking work automatically.  Otherwise the
        *loader* is called directly — the caller is responsible for
        freeing the returned object.

        The method is generic: the return type matches the loader's return
        type, preserving type information for callers that use typed
        protocol loaders (e.g. ``Callable[[], TTSModel]``).
        """
        if self.model_manager is not None:
            if name not in self.model_manager.registered_names:
                self.model_manager.register(name, loader=loader)
            return self.model_manager.get(name)  # type: ignore[no-any-return]
        return loader()

    def _resolve_vocals(self, state: PipelineState) -> Path:
        """Resolve the vocals file produced by the separation step."""
        return self.resolve_step_file(state, StepName.SEPARATE, "vocals")

    @property
    def step_dir(self) -> Path:
        """Convenience alias for the step working directory."""
        return self.work_dir
