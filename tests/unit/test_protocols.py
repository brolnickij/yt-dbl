"""Tests for yt_dbl.models.protocols — runtime_checkable Protocol contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

from yt_dbl.models.protocols import (
    AlignerModel,
    AlignerResultItem,
    STTModel,
    STTResult,
    TTSChunk,
    TTSModel,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

# ── TTSChunk ────────────────────────────────────────────────────────────────


class TestTTSChunk:
    def test_conforming_object(self) -> None:
        """An object with an .audio property satisfies TTSChunk."""

        class _Chunk:
            @property
            def audio(self) -> Any:
                return [0.0, 0.1]

        assert isinstance(_Chunk(), TTSChunk)

    def test_plain_mock_with_audio(self) -> None:
        mock = MagicMock()
        mock.audio = [0.0]
        assert isinstance(mock, TTSChunk)

    def test_missing_audio_fails(self) -> None:
        """An object without .audio does not satisfy TTSChunk."""

        class _NoAudio:
            pass

        assert not isinstance(_NoAudio(), TTSChunk)


# ── TTSModel ────────────────────────────────────────────────────────────────


class TestTTSModel:
    def test_conforming_class(self) -> None:
        class _FakeTTS:
            def generate(self, **kwargs: Any) -> Iterator[Any]:
                yield MagicMock(audio=[0.0])

        assert isinstance(_FakeTTS(), TTSModel)

    def test_mock_satisfies(self) -> None:
        class _Spec:
            def generate(self, **kwargs: Any) -> Any: ...

        mock = MagicMock(spec=_Spec)
        assert isinstance(mock, TTSModel)

    def test_missing_generate_fails(self) -> None:
        class _NoGenerate:
            pass

        assert not isinstance(_NoGenerate(), TTSModel)


# ── STTResult ───────────────────────────────────────────────────────────────


class TestSTTResult:
    def test_conforming_object(self) -> None:
        class _Result:
            @property
            def segments(self) -> list[dict[str, Any]] | None:
                return [{"start": 0.0, "end": 1.0}]

            @property
            def text(self) -> str:
                return "hello"

        assert isinstance(_Result(), STTResult)


# ── STTModel ────────────────────────────────────────────────────────────────


class TestSTTModel:
    def test_conforming_class(self) -> None:
        class _FakeSTT:
            def generate(self, *, audio: Any, **kwargs: Any) -> Any:
                return MagicMock(segments=[], text="")

        assert isinstance(_FakeSTT(), STTModel)


# ── AlignerResultItem ──────────────────────────────────────────────────────


class TestAlignerResultItem:
    def test_conforming_object(self) -> None:
        class _Item:
            @property
            def start_time(self) -> float:
                return 0.0

            @property
            def end_time(self) -> float:
                return 1.0

            @property
            def text(self) -> str:
                return "word"

        assert isinstance(_Item(), AlignerResultItem)

    def test_missing_end_time_fails(self) -> None:
        class _Partial:
            @property
            def start_time(self) -> float:
                return 0.0

            @property
            def text(self) -> str:
                return "word"

        assert not isinstance(_Partial(), AlignerResultItem)


# ── AlignerModel ───────────────────────────────────────────────────────────


class TestAlignerModel:
    def test_conforming_class(self) -> None:
        class _FakeAligner:
            def generate(self, *, audio: Any, text: str, language: str) -> Any:
                return []

        assert isinstance(_FakeAligner(), AlignerModel)
