"""Protocol types for ML models used in the pipeline.

Provides structural typing for third-party model objects (mlx-audio)
that are loaded at runtime via lazy imports.  Using protocols instead
of ``Any`` gives mypy enough information to catch method-name typos
and wrong argument types in pipeline steps, without importing heavy
ML dependencies at module level.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

__all__ = [
    "AlignerModel",
    "AlignerResultItem",
    "STTModel",
    "STTResult",
    "TTSChunk",
    "TTSModel",
]


# ── TTS (Qwen3-TTS via mlx-audio) ──────────────────────────────────────────


@runtime_checkable
class TTSChunk(Protocol):
    """A single chunk returned by ``TTSModel.generate``."""

    @property
    def audio(self) -> Any:
        """Raw audio tensor (mlx.core.array)."""
        ...


@runtime_checkable
class TTSModel(Protocol):
    """Text-to-speech model with optional voice cloning.

    Expected implementation: ``mlx_audio.tts.utils.load_model`` products.
    """

    def generate(self, **kwargs: Any) -> Any:
        """Yield audio chunks for the given text.

        Typical kwargs: ``text``, ``ref_audio``, ``ref_text``,
        ``temperature``, ``top_k``, ``top_p``, ``repetition_penalty``,
        ``lang_code``.
        """
        ...


# ── STT / ASR (VibeVoice-ASR via mlx-audio) ────────────────────────────────


@runtime_checkable
class STTResult(Protocol):
    """Result returned by ``STTModel.generate``."""

    @property
    def segments(self) -> list[dict[str, Any]] | None:
        """Parsed segments (may be ``None`` for raw-text output)."""
        ...

    @property
    def text(self) -> str:
        """Raw transcription text (fallback when segments is None)."""
        ...


@runtime_checkable
class STTModel(Protocol):
    """Automatic speech recognition model with diarization.

    Expected implementation: ``mlx_audio.stt.utils.load`` with
    VibeVoice-ASR weights.
    """

    def generate(self, *, audio: Any, **kwargs: Any) -> STTResult:
        """Transcribe audio, returning segments with speaker labels.

        Typical kwargs: ``max_tokens``, ``temperature``.
        """
        ...


# ── Forced Aligner (Qwen3-ForcedAligner via mlx-audio) ─────────────────────


@runtime_checkable
class AlignerResultItem(Protocol):
    """A single aligned word returned by the aligner."""

    @property
    def start_time(self) -> float: ...

    @property
    def end_time(self) -> float: ...

    @property
    def text(self) -> str: ...


@runtime_checkable
class AlignerModel(Protocol):
    """Word-level forced alignment model.

    Expected implementation: ``mlx_audio.stt.utils.load`` with
    Qwen3-ForcedAligner weights.
    """

    def generate(self, *, audio: Any, text: str, language: str) -> Any:
        """Align text to audio, returning items with word timestamps."""
        ...
