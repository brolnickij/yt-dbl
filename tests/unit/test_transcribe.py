"""Tests for yt_dbl.pipeline.transcribe — transcription step (mocked)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest

from yt_dbl.config import Settings
from yt_dbl.pipeline.base import StepValidationError
from yt_dbl.pipeline.transcribe import (
    _ALIGNER_LANGUAGE_MAP,
    SEGMENTS_FILE,
    TranscribeStep,
    _reference_score,
)
from yt_dbl.schemas import PipelineState, Segment, Speaker, StepName, StepStatus, Word

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


# ── Fake model outputs ──────────────────────────────────────────────────────


def _fake_asr_result(
    segments: list[dict[str, Any]] | None = None,
) -> Any:
    """Create a fake STTOutput-like object."""

    @dataclass
    class _FakeSTTOutput:
        text: str
        segments: list[dict[str, Any]] | None

    if segments is None:
        segs: list[dict[str, Any]] = [
            {"start": 0.0, "end": 3.5, "speaker_id": 0, "text": "Hello world."},
            {"start": 4.0, "end": 8.0, "speaker_id": 0, "text": "How are you?"},
            {"start": 8.5, "end": 11.0, "speaker_id": 1, "text": "I'm fine."},
        ]
    else:
        segs = segments

    # Build full_text from whichever text key is present
    texts: list[str] = []
    for s in segs:
        for key in ("text", "Content"):
            if key in s:
                texts.append(str(s[key]))
                break
    full_text = " ".join(texts)
    return _FakeSTTOutput(text=full_text, segments=segs)


@dataclass(frozen=True)
class _FakeAlignItem:
    text: str
    start_time: float
    end_time: float


@dataclass(frozen=True)
class _FakeAlignResult:
    items: list[_FakeAlignItem]

    def __iter__(self) -> Iterator[_FakeAlignItem]:
        return iter(self.items)


def _make_align_result(words: list[str], start: float, duration: float) -> _FakeAlignResult:
    """Build a fake ForcedAlignResult from word list."""
    gap = duration / max(len(words), 1)
    items = []
    t = start
    for w in words:
        items.append(_FakeAlignItem(text=w, start_time=round(t, 3), end_time=round(t + gap, 3)))
        t += gap
    return _FakeAlignResult(items=items)


# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_step(tmp_path: Path) -> tuple[TranscribeStep, Settings, PipelineState]:
    cfg = Settings(work_dir=tmp_path / "work")
    step_dir = cfg.step_dir("test123", "03_transcribe")
    step = TranscribeStep(settings=cfg, work_dir=step_dir)
    state = PipelineState(video_id="test123", url="https://example.com")

    # Prefill download + separate
    dl = state.get_step(StepName.DOWNLOAD)
    dl.status = StepStatus.COMPLETED
    dl.outputs = {"video": "video.mp4", "audio": "audio.wav"}
    dl_dir = cfg.step_dir("test123", "01_download")
    (dl_dir / "audio.wav").write_bytes(b"fake")

    sep = state.get_step(StepName.SEPARATE)
    sep.status = StepStatus.COMPLETED
    sep.outputs = {"vocals": "vocals.wav", "background": "background.wav"}
    sep_dir = cfg.step_dir("test123", "02_separate")
    (sep_dir / "vocals.wav").write_bytes(b"fake-vocals")
    (sep_dir / "background.wav").write_bytes(b"fake-bg")

    return step, cfg, state


# ── Validation tests ────────────────────────────────────────────────────────


class TestTranscribeStepValidation:
    def test_validate_missing_vocals(self, tmp_path: Path) -> None:
        step, _, state = _make_step(tmp_path)
        state.get_step(StepName.SEPARATE).outputs = {}
        with pytest.raises(StepValidationError, match="No vocals file"):
            step.validate_inputs(state)

    def test_validate_ok(self, tmp_path: Path) -> None:
        step, _, state = _make_step(tmp_path)
        step.validate_inputs(state)  # should not raise


# ── ASR normalisation tests ─────────────────────────────────────────────────


class TestNormaliseASRSegments:
    def test_standard_keys(self) -> None:
        result = _fake_asr_result()
        segs = TranscribeStep._normalise_asr_segments(result)
        assert len(segs) == 3
        assert segs[0]["text"] == "Hello world."
        assert segs[0]["start"] == 0.0
        assert segs[0]["end"] == 3.5
        assert segs[0]["speaker_id"] == 0

    def test_vibevoice_json_keys(self) -> None:
        """VibeVoice may use 'Start', 'End', 'Speaker', 'Content' keys."""
        result = _fake_asr_result(
            [
                {"Start": 0.0, "End": 5.2, "Speaker": 0, "Content": "Hello everyone."},
                {"Start": 5.5, "End": 9.8, "Speaker": 1, "Content": "Thanks for joining."},
            ]
        )
        segs = TranscribeStep._normalise_asr_segments(result)
        assert len(segs) == 2
        assert segs[0]["text"] == "Hello everyone."
        assert segs[1]["speaker_id"] == 1

    def test_start_time_end_time_keys(self) -> None:
        """Some results use 'start_time' / 'end_time'."""
        result = _fake_asr_result(
            [{"start_time": 1.0, "end_time": 2.0, "speaker_id": 0, "text": "OK"}]
        )
        segs = TranscribeStep._normalise_asr_segments(result)
        assert segs[0]["start"] == 1.0
        assert segs[0]["end"] == 2.0

    def test_empty_segments(self) -> None:
        result = _fake_asr_result([])
        segs = TranscribeStep._normalise_asr_segments(result)
        assert segs == []

    def test_fallback_json_from_text(self) -> None:
        """When .segments is None, try parsing JSON from .text."""

        @dataclass
        class _FakeResult:
            text: str
            segments: None = None

        result = _FakeResult(text='[{"start": 0, "end": 1, "speaker_id": 0, "text": "Hello"}]')
        segs = TranscribeStep._normalise_asr_segments(result)
        assert len(segs) == 1
        assert segs[0]["text"] == "Hello"

    def test_incomplete_segment_skipped(self) -> None:
        """Segments missing required fields are dropped."""
        result = _fake_asr_result(
            [{"start": 0.0}]  # missing end and text
        )
        segs = TranscribeStep._normalise_asr_segments(result)
        assert segs == []

    def test_default_speaker_id(self) -> None:
        """Segments without speaker get speaker_id=0."""
        result = _fake_asr_result([{"start": 0.0, "end": 1.0, "text": "No speaker"}])
        segs = TranscribeStep._normalise_asr_segments(result)
        assert segs[0]["speaker_id"] == 0


# ── Alignment tests ─────────────────────────────────────────────────────────


class TestAlignSegment:
    def test_align_produces_words(self) -> None:
        aligner = MagicMock()
        aligner.generate.return_value = _make_align_result(["Hello", "world"], 0.0, 3.5)

        words = TranscribeStep._align_segment(
            aligner,
            audio=MagicMock(),
            seg={"text": "Hello world", "start": 0.0, "end": 3.5, "speaker_id": 0},
            language="English",
        )

        assert len(words) == 2
        assert all(isinstance(w, Word) for w in words)
        assert words[0].text == "Hello"
        assert words[1].text == "world"

    def test_align_empty_text(self) -> None:
        aligner = MagicMock()
        words = TranscribeStep._align_segment(
            aligner,
            audio=MagicMock(),
            seg={"text": "  ", "start": 0.0, "end": 1.0, "speaker_id": 0},
            language="English",
        )
        assert words == []
        aligner.generate.assert_not_called()

    def test_align_fallback_on_error(self) -> None:
        aligner = MagicMock()
        aligner.generate.side_effect = RuntimeError("model failed")

        words = TranscribeStep._align_segment(
            aligner,
            audio=MagicMock(),
            seg={"text": "Fallback text", "start": 1.0, "end": 2.0, "speaker_id": 0},
            language="English",
        )

        assert len(words) == 1
        assert words[0].text == "Fallback text"
        assert words[0].confidence == 0.5

    def test_align_applies_time_offset(self) -> None:
        """Word timestamps are shifted by time_offset (audio slice → absolute)."""
        aligner = MagicMock()
        # Aligner returns timestamps relative to slice start (0.0)
        aligner.generate.return_value = _make_align_result(["Hello", "world"], 0.0, 3.0)

        offset = 10.0
        words = TranscribeStep._align_segment(
            aligner,
            audio=MagicMock(),
            seg={"text": "Hello world", "start": 10.0, "end": 13.0, "speaker_id": 0},
            language="English",
            time_offset=offset,
        )

        assert len(words) == 2
        # All timestamps should be shifted by offset
        assert words[0].start == pytest.approx(0.0 + offset)
        assert words[1].start == pytest.approx(1.5 + offset)


# ── Language detection tests ────────────────────────────────────────────────


class TestLanguageDetection:
    def test_english(self) -> None:
        segs = [{"text": "Hello world, this is a test."}]
        assert TranscribeStep._detect_language(segs) == "en"

    def test_russian(self) -> None:
        segs = [{"text": "Привет мир, это тест."}]
        assert TranscribeStep._detect_language(segs) == "ru"

    def test_chinese(self) -> None:
        segs = [{"text": "你好世界"}]
        assert TranscribeStep._detect_language(segs) == "zh"

    def test_japanese(self) -> None:
        segs = [{"text": "こんにちは世界"}]
        assert TranscribeStep._detect_language(segs) == "ja"

    def test_korean(self) -> None:
        segs = [{"text": "안녕하세요 세계"}]
        assert TranscribeStep._detect_language(segs) == "ko"

    def test_empty(self) -> None:
        assert TranscribeStep._detect_language([]) == "en"


# ── Reference score tests ───────────────────────────────────────────────────


class TestReferenceScore:
    def test_high_confidence_long_segment(self) -> None:
        seg = Segment(
            id=0,
            text="Hello world",
            start=0.0,
            end=5.0,
            speaker="SPEAKER_00",
            words=[
                Word(text="Hello", start=0.0, end=2.5, confidence=1.0),
                Word(text="world", start=2.5, end=5.0, confidence=1.0),
            ],
        )
        # score = 1.0 * min(5.0, 8.0) = 5.0
        assert _reference_score(seg) == pytest.approx(5.0)

    def test_fallback_alignment_lowers_score(self) -> None:
        seg = Segment(
            id=0,
            text="Fallback",
            start=0.0,
            end=6.0,
            speaker="SPEAKER_00",
            words=[Word(text="Fallback", start=0.0, end=6.0, confidence=0.5)],
        )
        # score = 0.5 * 6.0 = 3.0
        assert _reference_score(seg) == pytest.approx(3.0)

    def test_no_words_defaults_to_low_confidence(self) -> None:
        seg = Segment(id=0, text="No words", start=0.0, end=4.0, speaker="SPEAKER_00")
        # score = 0.5 * 4.0 = 2.0
        assert _reference_score(seg) == pytest.approx(2.0)

    def test_short_segment_penalised(self) -> None:
        seg = Segment(
            id=0,
            text="Hi",
            start=0.0,
            end=2.0,
            speaker="SPEAKER_00",
            words=[Word(text="Hi", start=0.0, end=2.0, confidence=1.0)],
        )
        # score = 1.0 * 2.0 * 0.5(penalty) = 1.0
        assert _reference_score(seg) == pytest.approx(1.0)

    def test_very_long_segment_capped(self) -> None:
        seg = Segment(
            id=0,
            text="Very long",
            start=0.0,
            end=20.0,
            speaker="SPEAKER_00",
            words=[Word(text="Very long", start=0.0, end=20.0, confidence=1.0)],
        )
        # score = 1.0 * 8.0(capped) = 8.0
        assert _reference_score(seg) == pytest.approx(8.0)


# ── Speaker extraction tests ────────────────────────────────────────────────


class TestExtractSpeakers:
    def test_basic(self) -> None:
        segments = [
            Segment(id=0, text="A", start=0.0, end=3.0, speaker="SPEAKER_00"),
            Segment(id=1, text="B", start=3.0, end=5.0, speaker="SPEAKER_01"),
            Segment(id=2, text="C", start=5.0, end=9.0, speaker="SPEAKER_00"),
        ]
        speakers = TranscribeStep._extract_speakers(segments)

        assert len(speakers) == 2
        sp0 = next(s for s in speakers if s.id == "SPEAKER_00")
        sp1 = next(s for s in speakers if s.id == "SPEAKER_01")
        assert sp0.total_duration == pytest.approx(7.0, abs=0.01)
        assert sp1.total_duration == pytest.approx(2.0, abs=0.01)

    def test_reference_prefers_aligned_over_longest(self) -> None:
        """A well-aligned segment scores higher than a longer fallback one."""
        segments = [
            Segment(
                id=0,
                text="Well aligned segment",
                start=0.0,
                end=5.0,
                speaker="SPEAKER_00",
                words=[
                    Word(text="Well", start=0.0, end=1.0, confidence=1.0),
                    Word(text="aligned", start=1.0, end=2.5, confidence=1.0),
                    Word(text="segment", start=2.5, end=5.0, confidence=1.0),
                ],
            ),
            Segment(
                id=1,
                text="Longer but alignment failed completely",
                start=5.0,
                end=15.0,
                speaker="SPEAKER_00",
                words=[
                    Word(
                        text="Longer but alignment failed completely",
                        start=5.0,
                        end=15.0,
                        confidence=0.5,
                    ),
                ],
            ),
        ]
        speakers = TranscribeStep._extract_speakers(segments)
        # Clean: 1.0 * 5.0 = 5.0  |  Fallback: 0.5 * 8.0(capped) = 4.0
        assert speakers[0].reference_start == 0.0
        assert speakers[0].reference_end == 5.0

    def test_reference_penalises_very_short(self) -> None:
        """Segments shorter than 3s get a penalty even with perfect confidence."""
        segments = [
            Segment(
                id=0,
                text="Tiny",
                start=0.0,
                end=1.5,
                speaker="SPEAKER_00",
                words=[Word(text="Tiny", start=0.0, end=1.5, confidence=1.0)],
            ),
            Segment(
                id=1,
                text="Decent length ok",
                start=2.0,
                end=6.0,
                speaker="SPEAKER_00",
                words=[
                    Word(text="Decent", start=2.0, end=3.5, confidence=1.0),
                    Word(text="length", start=3.5, end=4.5, confidence=1.0),
                    Word(text="ok", start=4.5, end=6.0, confidence=1.0),
                ],
            ),
        ]
        speakers = TranscribeStep._extract_speakers(segments)
        # Tiny: 1.0 * 1.5 * 0.5(penalty) = 0.75  |  Decent: 1.0 * 4.0 = 4.0
        assert speakers[0].reference_start == 2.0
        assert speakers[0].reference_end == 6.0


# ── Persistence (save/load) tests ──────────────────────────────────────────


class TestPersistence:
    def test_save_and_load(self, tmp_path: Path) -> None:
        segments = [
            Segment(
                id=0,
                text="Hello",
                start=0.0,
                end=1.0,
                speaker="SPEAKER_00",
                words=[Word(text="Hello", start=0.0, end=1.0)],
            ),
        ]
        speakers = [Speaker(id="SPEAKER_00", total_duration=1.0)]

        path = tmp_path / SEGMENTS_FILE
        TranscribeStep._save(path, segments, speakers)

        assert path.exists()
        data = json.loads(path.read_text())
        assert len(data["segments"]) == 1
        assert len(data["speakers"]) == 1

    def test_load_cached(self, tmp_path: Path) -> None:
        step, _cfg, state = _make_step(tmp_path)
        segments = [
            Segment(id=0, text="Cached", start=0.0, end=1.0, speaker="SPEAKER_00"),
        ]
        speakers = [Speaker(id="SPEAKER_00", total_duration=1.0)]

        path = step.step_dir / SEGMENTS_FILE
        TranscribeStep._save(path, segments, speakers)

        loaded_state = TranscribeStep._load_cached(state, path)

        assert len(loaded_state.segments) == 1
        assert loaded_state.segments[0].text == "Cached"
        assert len(loaded_state.speakers) == 1


# ── Full run (mocked models) ───────────────────────────────────────────────


class TestTranscribeStepRun:
    @patch("yt_dbl.pipeline.transcribe.TranscribeStep._run_alignment")
    @patch("yt_dbl.pipeline.transcribe.TranscribeStep._run_asr")
    def test_run_success(
        self,
        mock_asr: MagicMock,
        mock_align: MagicMock,
        tmp_path: Path,
    ) -> None:
        step, _, state = _make_step(tmp_path)

        mock_asr.return_value = [
            {"start": 0.0, "end": 3.0, "speaker_id": 0, "text": "Hello"},
            {"start": 4.0, "end": 7.0, "speaker_id": 1, "text": "World"},
        ]
        mock_align.return_value = [
            Segment(
                id=0,
                text="Hello",
                start=0.0,
                end=3.0,
                speaker="SPEAKER_00",
                words=[Word(text="Hello", start=0.0, end=3.0)],
            ),
            Segment(
                id=1,
                text="World",
                start=4.0,
                end=7.0,
                speaker="SPEAKER_01",
                words=[Word(text="World", start=4.0, end=7.0)],
            ),
        ]

        state = step.run(state)

        assert len(state.segments) == 2
        assert len(state.speakers) == 2
        assert state.get_step(StepName.TRANSCRIBE).outputs["segments"] == SEGMENTS_FILE
        assert (step.step_dir / SEGMENTS_FILE).exists()

    @patch("yt_dbl.pipeline.transcribe.TranscribeStep._run_alignment")
    @patch("yt_dbl.pipeline.transcribe.TranscribeStep._run_asr")
    def test_run_idempotent(
        self,
        mock_asr: MagicMock,
        mock_align: MagicMock,
        tmp_path: Path,
    ) -> None:
        step, _, state = _make_step(tmp_path)

        # Pre-create cached result
        segments = [Segment(id=0, text="Cached", start=0.0, end=1.0, speaker="SPEAKER_00")]
        speakers = [Speaker(id="SPEAKER_00", total_duration=1.0)]
        TranscribeStep._save(step.step_dir / SEGMENTS_FILE, segments, speakers)

        state = step.run(state)

        # ASR and alignment should NOT be called
        mock_asr.assert_not_called()
        mock_align.assert_not_called()
        assert state.segments[0].text == "Cached"


# ── Config tests ────────────────────────────────────────────────────────────


class TestTranscriptionConfig:
    def test_default_asr_model(self) -> None:
        cfg = Settings()
        assert "VibeVoice" in cfg.transcription_asr_model

    def test_default_aligner_model(self) -> None:
        cfg = Settings()
        assert "ForcedAligner" in cfg.transcription_aligner_model

    def test_default_max_tokens(self) -> None:
        cfg = Settings()
        assert cfg.transcription_max_tokens == 8192

    def test_default_temperature(self) -> None:
        cfg = Settings()
        assert cfg.transcription_temperature == 0.0

    def test_custom_model_via_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("YT_DBL_TRANSCRIPTION_ASR_MODEL", "custom/model")
        cfg = Settings()
        assert cfg.transcription_asr_model == "custom/model"


# ── Language map ────────────────────────────────────────────────────────────


class TestLanguageMap:
    def test_common_languages_present(self) -> None:
        for code in ("en", "ru", "zh", "ja", "ko", "de", "fr", "es"):
            assert code in _ALIGNER_LANGUAGE_MAP

    def test_values_are_capitalized(self) -> None:
        for lang in _ALIGNER_LANGUAGE_MAP.values():
            assert lang[0].isupper()
