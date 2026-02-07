"""Tests for yt_dbl.pipeline.transcribe — transcription step (mocked)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest

from yt_dbl.config import Settings
from yt_dbl.pipeline.base import StepValidationError, TranscriptionError
from yt_dbl.pipeline.transcribe import (
    SEGMENTS_FILE,
    TranscribeStep,
    _parse_timestamp,
    _recover_partial_json,
    _reference_score,
)
from yt_dbl.schemas import STEP_DIRS, PipelineState, Segment, Speaker, StepName, StepStatus, Word
from yt_dbl.utils.languages import ALIGNER_LANGUAGE_MAP, TTS_LANG_MAP

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
    step_dir = cfg.step_dir("test123", STEP_DIRS[StepName.TRANSCRIBE])
    step = TranscribeStep(settings=cfg, work_dir=step_dir)
    state = PipelineState(video_id="test123", url="https://example.com")

    # Prefill download + separate
    dl = state.get_step(StepName.DOWNLOAD)
    dl.status = StepStatus.COMPLETED
    dl.outputs = {"video": "video.mp4", "audio": "audio.wav"}
    dl_dir = cfg.step_dir("test123", STEP_DIRS[StepName.DOWNLOAD])
    (dl_dir / "audio.wav").write_bytes(b"fake")

    sep = state.get_step(StepName.SEPARATE)
    sep.status = StepStatus.COMPLETED
    sep.outputs = {"vocals": "vocals.wav", "background": "background.wav"}
    sep_dir = cfg.step_dir("test123", STEP_DIRS[StepName.SEPARATE])
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

    def test_string_speaker_id_falls_back_to_default(self) -> None:
        """Non-numeric speaker_id (e.g. 'SPEAKER_00') falls back to 0."""
        result = _fake_asr_result(
            [{"start": 0.0, "end": 2.0, "speaker_id": "SPEAKER_00", "text": "Hello"}]
        )
        segs = TranscribeStep._normalise_asr_segments(result)
        assert len(segs) == 1
        assert segs[0]["speaker_id"] == 0

    def test_numeric_string_speaker_id_casts_ok(self) -> None:
        """Speaker ID like '2' (string of a number) should cast to int."""
        result = _fake_asr_result([{"start": 0.0, "end": 1.0, "speaker_id": "2", "text": "Hi"}])
        segs = TranscribeStep._normalise_asr_segments(result)
        assert segs[0]["speaker_id"] == 2


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
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Hello world, this is a test.", "en"),
            ("Привет мир, это тест.", "ru"),
            ("你好世界", "zh"),
            ("こんにちは世界", "ja"),
            ("안녕하세요 세계", "ko"),
            ("مرحبا بالعالم", "ar"),
            ("नमस्ते दुनिया", "hi"),
            ("สวัสดีชาวโลก", "th"),
        ],
        ids=["english", "russian", "chinese", "japanese", "korean", "arabic", "hindi", "thai"],
    )
    def test_detects_language(self, text: str, expected: str) -> None:
        assert TranscribeStep._detect_language([{"text": text}]) == expected

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

    def test_save_includes_source_language(self, tmp_path: Path) -> None:
        """source_language is stored in segments.json."""
        segments = [Segment(id=0, text="Привет", start=0.0, end=1.0, speaker="SPEAKER_00")]
        speakers = [Speaker(id="SPEAKER_00", total_duration=1.0)]

        path = tmp_path / SEGMENTS_FILE
        TranscribeStep._save(path, segments, speakers, source_language="ru")

        data = json.loads(path.read_text())
        assert data["source_language"] == "ru"

    def test_load_cached_restores_source_language(self, tmp_path: Path) -> None:
        """_load_cached restores source_language from segments.json."""
        step, _cfg, state = _make_step(tmp_path)
        segments = [Segment(id=0, text="Hola", start=0.0, end=1.0, speaker="SPEAKER_00")]
        speakers = [Speaker(id="SPEAKER_00", total_duration=1.0)]

        path = step.step_dir / SEGMENTS_FILE
        TranscribeStep._save(path, segments, speakers, source_language="es")

        loaded = TranscribeStep._load_cached(state, path)
        assert loaded.source_language == "es"

    def test_load_cached_handles_legacy_file(self, tmp_path: Path) -> None:
        """Old segments.json without source_language doesn't crash."""
        step, _cfg, state = _make_step(tmp_path)
        segments = [Segment(id=0, text="Hi", start=0.0, end=1.0, speaker="SPEAKER_00")]
        speakers = [Speaker(id="SPEAKER_00", total_duration=1.0)]

        path = step.step_dir / SEGMENTS_FILE
        # Write without source_language (legacy format)
        data = {
            "segments": [s.model_dump() for s in segments],
            "speakers": [s.model_dump() for s in speakers],
        }
        path.write_text(json.dumps(data))

        loaded = TranscribeStep._load_cached(state, path)
        assert loaded.source_language == ""  # remains default


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

    @patch("yt_dbl.pipeline.transcribe.TranscribeStep._run_asr")
    def test_run_asr_failure_raises_transcription_error(
        self,
        mock_asr: MagicMock,
        tmp_path: Path,
    ) -> None:
        """When _run_asr raises, the error is a TranscriptionError."""
        step, _, state = _make_step(tmp_path)
        mock_asr.side_effect = TranscriptionError("ASR model failed: OOM")

        with pytest.raises(TranscriptionError, match="ASR model failed"):
            step.run(state)


# ── Config tests ────────────────────────────────────────────────────────────


class TestTranscriptionConfig:
    def test_default_asr_model(self) -> None:
        cfg = Settings()
        assert "VibeVoice" in cfg.transcription_asr_model

    def test_default_aligner_model(self) -> None:
        cfg = Settings()
        assert "ForcedAligner" in cfg.transcription_aligner_model

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
            assert code in ALIGNER_LANGUAGE_MAP

    def test_values_are_capitalized(self) -> None:
        for lang in ALIGNER_LANGUAGE_MAP.values():
            assert lang[0].isupper()


class TestTTSLangMap:
    def test_common_languages_present(self) -> None:
        for code in ("en", "ru", "zh", "ja", "de", "fr", "es"):
            assert code in TTS_LANG_MAP

    def test_values_are_lowercase(self) -> None:
        for lang in TTS_LANG_MAP.values():
            assert lang == lang.lower()

    def test_subset_of_aligner_map(self) -> None:
        """Every TTS language should also be in the aligner map."""
        for code in TTS_LANG_MAP:
            assert code in ALIGNER_LANGUAGE_MAP


# ── Chunk merging tests ─────────────────────────────────────────────────────


class TestMergeChunkSegments:
    """Tests for TranscribeStep._merge_chunk_segments."""

    def test_single_chunk_passthrough(self) -> None:
        segs = [{"start": 0.0, "end": 3.0, "speaker_id": 0, "text": "Hi"}]
        result = TranscribeStep._merge_chunk_segments([(0.0, 3300.0, segs)], 120.0)
        assert result == segs

    def test_empty_input(self) -> None:
        assert TranscribeStep._merge_chunk_segments([], 120.0) == []

    def test_two_chunks_no_duplicates(self) -> None:
        """Segments in overlap zone are deduplicated by midpoint rule."""
        overlap_sec = 120.0
        # Chunk 0: [0, 3300], Chunk 1: [3180, 6480]
        # Overlap zone: [3180, 3300], midpoint = 3180 + 60 = 3240
        seg_a1 = {"start": 100.0, "end": 105.0, "speaker_id": 0, "text": "A1"}
        seg_a2 = {"start": 3200.0, "end": 3210.0, "speaker_id": 0, "text": "A2"}  # in overlap
        seg_a3 = {"start": 3250.0, "end": 3260.0, "speaker_id": 0, "text": "A3"}  # past midpoint

        seg_b1 = {"start": 3200.0, "end": 3210.0, "speaker_id": 0, "text": "B1"}  # before midpoint
        seg_b2 = {"start": 3250.0, "end": 3260.0, "speaker_id": 0, "text": "B2"}  # past midpoint
        seg_b3 = {"start": 4000.0, "end": 4010.0, "speaker_id": 0, "text": "B3"}

        chunks = [
            (0.0, 3300.0, [seg_a1, seg_a2, seg_a3]),
            (3180.0, 6480.0, [seg_b1, seg_b2, seg_b3]),
        ]
        result = TranscribeStep._merge_chunk_segments(chunks, overlap_sec)

        texts = [s["text"] for s in result]
        # A1 (< midpoint 3240) from chunk 0
        # A2 (start=3200 < 3240) from chunk 0
        # A3 (start=3250 >= 3240) NOT from chunk 0 (hi=3240)
        # B1 (start=3200 < 3240) NOT from chunk 1 (lo=3240)
        # B2 (start=3250 >= 3240) from chunk 1
        # B3 from chunk 1
        assert texts == ["A1", "A2", "B2", "B3"]

    def test_three_chunks(self) -> None:
        overlap_sec = 120.0
        # Chunk 0: [0, 3300], Chunk 1: [3180, 6480], Chunk 2: [6360, 9660]
        # Mid 0-1 = 3240, Mid 1-2 = 6420
        chunks = [
            (0.0, 3300.0, [{"start": 1000.0, "end": 1010.0, "speaker_id": 0, "text": "C0"}]),
            (3180.0, 6480.0, [{"start": 5000.0, "end": 5010.0, "speaker_id": 0, "text": "C1"}]),
            (6360.0, 9660.0, [{"start": 8000.0, "end": 8010.0, "speaker_id": 0, "text": "C2"}]),
        ]
        result = TranscribeStep._merge_chunk_segments(chunks, overlap_sec)
        assert [s["text"] for s in result] == ["C0", "C1", "C2"]


# ── Speaker reconciliation tests ────────────────────────────────────────────


class TestReconcileChunkSpeakers:
    """Tests for TranscribeStep._reconcile_chunk_speakers."""

    def test_matching_two_speakers(self) -> None:
        """Speakers in overlap zone are correctly matched by temporal overlap."""
        # Previous chunk (global IDs already assigned): speaker 0 and 1
        prev = [
            {"start": 3180.0, "end": 3220.0, "speaker_id": 0, "text": "prev-spk0"},
            {"start": 3220.0, "end": 3300.0, "speaker_id": 1, "text": "prev-spk1"},
        ]
        # Current chunk (local IDs): speaker 0 and 1 (swapped)
        curr = [
            {"start": 3180.0, "end": 3230.0, "speaker_id": 1, "text": "curr-spk1"},
            {"start": 3230.0, "end": 3300.0, "speaker_id": 0, "text": "curr-spk0"},
            {"start": 3400.0, "end": 3500.0, "speaker_id": 0, "text": "outside"},
            {"start": 3500.0, "end": 3600.0, "speaker_id": 1, "text": "outside2"},
        ]
        mapping, gmax = TranscribeStep._reconcile_chunk_speakers(
            prev,
            curr,
            overlap_start=3180.0,
            overlap_end=3300.0,
            global_max_speaker=1,
        )
        # curr speaker 1 overlaps with prev speaker 0 (3180-3220 overlap)
        # curr speaker 0 overlaps with prev speaker 1 (3230-3300 overlap)
        assert mapping[1] == 0
        assert mapping[0] == 1
        assert gmax == 1  # no new speakers

    def test_no_overlap_data(self) -> None:
        """When no segments fall in overlap zone, assign fresh IDs."""
        prev = [{"start": 100.0, "end": 200.0, "speaker_id": 0, "text": "far away"}]
        curr = [
            {"start": 5000.0, "end": 5100.0, "speaker_id": 0, "text": "also far"},
            {"start": 5100.0, "end": 5200.0, "speaker_id": 1, "text": "also far"},
        ]
        mapping, gmax = TranscribeStep._reconcile_chunk_speakers(
            prev,
            curr,
            overlap_start=3180.0,
            overlap_end=3300.0,
            global_max_speaker=1,
        )
        # No overlap data → fresh IDs: 2 and 3
        assert mapping[0] == 2
        assert mapping[1] == 3
        assert gmax == 3

    def test_new_speaker_in_later_chunk(self) -> None:
        """A speaker appearing only in the current chunk gets a fresh ID."""
        prev = [
            {"start": 3180.0, "end": 3300.0, "speaker_id": 0, "text": "prev"},
        ]
        curr = [
            {"start": 3180.0, "end": 3300.0, "speaker_id": 0, "text": "match"},
            {"start": 3400.0, "end": 3500.0, "speaker_id": 1, "text": "new speaker"},
        ]
        mapping, gmax = TranscribeStep._reconcile_chunk_speakers(
            prev,
            curr,
            overlap_start=3180.0,
            overlap_end=3300.0,
            global_max_speaker=0,
        )
        assert mapping[0] == 0  # matched to prev speaker 0
        assert mapping[1] == 1  # new global ID
        assert gmax == 1

    def test_hungarian_optimal_over_greedy(self) -> None:
        """Hungarian algorithm finds optimal matching where greedy would fail.

        Overlap matrix (seconds):
                     prev0   prev1
            curr 0:   51      50
            curr 1:   50       0

        Greedy picks curr 0→prev 0 (best single=51), leaving curr 1
        unmatched (prev 0 taken, prev 1 has zero overlap with curr 1).
        Total matched overlap: 51, one speaker unmapped.

        Hungarian picks curr 0→prev 1 (50) + curr 1→prev 0 (50) = 100,
        mapping both speakers optimally.
        """
        prev = [
            # prev speaker 0: spans entire overlap zone (3180–3300)
            {"start": 3180.0, "end": 3300.0, "speaker_id": 0, "text": "prev-spk0"},
            # prev speaker 1: first 50s of overlap zone (3180–3230)
            {"start": 3180.0, "end": 3230.0, "speaker_id": 1, "text": "prev-spk1"},
        ]
        curr = [
            # curr speaker 0: 3179–3231 → overlaps prev0 by 51s, prev1 by 50s
            {"start": 3179.0, "end": 3231.0, "speaker_id": 0, "text": "curr-spk0"},
            # curr speaker 1: 3250–3300 → overlaps prev0 by 50s, prev1 by 0s
            {"start": 3250.0, "end": 3300.0, "speaker_id": 1, "text": "curr-spk1"},
        ]
        mapping, gmax = TranscribeStep._reconcile_chunk_speakers(
            prev,
            curr,
            overlap_start=3180.0,
            overlap_end=3300.0,
            global_max_speaker=1,
        )
        # Hungarian optimal: curr 0→prev 1, curr 1→prev 0 (total overlap 100)
        # Greedy would pick: curr 0→prev 0 (51), curr 1 unmatched (total 51)
        assert mapping[0] == 1  # curr 0 → prev speaker 1
        assert mapping[1] == 0  # curr 1 → prev speaker 0
        assert gmax == 1  # no new speakers needed


# ── Chunking config tests ──────────────────────────────────────────────────


class TestChunkingConfig:
    def test_default_max_chunk_minutes(self) -> None:
        cfg = Settings()
        assert cfg.transcription_max_chunk_minutes == 30.0

    def test_default_chunk_overlap(self) -> None:
        cfg = Settings()
        assert cfg.transcription_chunk_overlap_minutes == 2.0

    def test_custom_chunk_settings_via_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("YT_DBL_TRANSCRIPTION_MAX_CHUNK_MINUTES", "45")
        monkeypatch.setenv("YT_DBL_TRANSCRIPTION_CHUNK_OVERLAP_MINUTES", "3")
        cfg = Settings()
        assert cfg.transcription_max_chunk_minutes == 45.0
        assert cfg.transcription_chunk_overlap_minutes == 3.0


# ── Chunk boundary computation tests ────────────────────────────────────────


class TestComputeChunkBoundaries:
    """Tests for TranscribeStep._compute_chunk_boundaries."""

    def test_single_chunk_when_short(self) -> None:
        """Audio shorter than max_chunk → one boundary spanning all."""
        # duration < max_chunk, but method is only called when duration > max_chunk
        # in practice.  Still, verify it returns a single chunk.
        result = TranscribeStep._compute_chunk_boundaries(300.0, 1800.0, 120.0)
        assert result == [(0.0, 300.0)]

    def test_two_overlapping_chunks(self) -> None:
        # duration=3600, chunk=1800, overlap=120 → step=1680
        # chunk 0: [0, 1800], chunk 1: [1680, 3480], chunk 2: [3360, 3600]
        result = TranscribeStep._compute_chunk_boundaries(3600.0, 1800.0, 120.0)
        assert len(result) == 3
        assert result[0] == (0.0, 1800.0)
        assert result[1] == (1680.0, 3480.0)
        assert result[2] == (3360.0, 3600.0)

    def test_three_chunks(self) -> None:
        # duration=5000, chunk=1800, overlap=120 → step=1680
        # chunk 0: [0, 1800], chunk 1: [1680, 3480], chunk 2: [3360, 5000]
        result = TranscribeStep._compute_chunk_boundaries(5000.0, 1800.0, 120.0)
        assert len(result) == 3
        assert result[0] == (0.0, 1800.0)
        assert result[1] == (1680.0, 3480.0)
        assert result[2] == (3360.0, 5000.0)

    def test_last_chunk_clamped_to_duration(self) -> None:
        """The final chunk end is clamped to duration, not max_chunk beyond."""
        result = TranscribeStep._compute_chunk_boundaries(2000.0, 1800.0, 120.0)
        assert result[-1][1] == 2000.0

    def test_overlap_equals_chunk_raises(self) -> None:
        """overlap == chunk → zero step → infinite loop guard."""
        with pytest.raises(TranscriptionError, match="must be less than"):
            TranscribeStep._compute_chunk_boundaries(7200.0, 300.0, 300.0)

    def test_overlap_exceeds_chunk_raises(self) -> None:
        """overlap > chunk → negative step → infinite loop guard."""
        with pytest.raises(TranscriptionError, match="must be less than"):
            TranscribeStep._compute_chunk_boundaries(7200.0, 300.0, 600.0)

    def test_small_step_produces_many_chunks(self) -> None:
        """chunk=300, overlap=270 → step=30 → many chunks but finite."""
        result = TranscribeStep._compute_chunk_boundaries(600.0, 300.0, 270.0)
        # step=30, so: 0,30,60,...,300 → last chunk covers to 600
        assert len(result) == 11
        assert result[0] == (0.0, 300.0)
        assert result[-1][1] == 600.0


# ── Timestamp parsing tests ─────────────────────────────────────────────────


class TestParseTimestamp:
    """Tests for _parse_timestamp helper."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0.0, 0.0),
            (12.5, 12.5),
            (100, 100.0),
            ("0", 0.0),
            ("12.5", 12.5),
            ("123.456", 123.456),
        ],
        ids=["float-zero", "float", "int", "str-zero", "str-float", "str-long"],
    )
    def test_numeric(self, value: Any, expected: float) -> None:
        assert _parse_timestamp(value) == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("0:00", 0.0),
            ("1:30", 90.0),
            ("55:00", 3300.0),
            ("0:05", 5.0),
            ("12:34", 754.0),
        ],
        ids=["zero", "one-min-thirty", "fifty-five-min", "five-sec", "twelve-thirty-four"],
    )
    def test_mm_ss(self, value: str, expected: float) -> None:
        assert _parse_timestamp(value) == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("0:00:00", 0.0),
            ("1:00:00", 3600.0),
            ("2:30:15", 9015.0),
            ("0:55:00", 3300.0),
        ],
        ids=["zero", "one-hour", "complex", "fifty-five-min"],
    )
    def test_hh_mm_ss(self, value: str, expected: float) -> None:
        assert _parse_timestamp(value) == pytest.approx(expected)

    @pytest.mark.parametrize(
        "value",
        [None, "", "abc", "12:ab", [], {}],
        ids=["none", "empty", "alpha", "bad-mmss", "list", "dict"],
    )
    def test_unparseable(self, value: Any) -> None:
        assert _parse_timestamp(value) is None

    def test_normalise_segment_with_mmss_timestamps(self) -> None:
        """Full integration: _normalise_asr_segments handles MM:SS timestamps."""
        result = _fake_asr_result(
            [
                {
                    "Start time": "1:30",
                    "End time": "2:00",
                    "Speaker ID": "0",
                    "Content": "Hello everyone.",
                },
            ]
        )
        segs = TranscribeStep._normalise_asr_segments(result)
        assert len(segs) == 1
        assert segs[0]["start"] == pytest.approx(90.0)
        assert segs[0]["end"] == pytest.approx(120.0)


# ── Partial JSON recovery tests ──────────────────────────────────────────────


class TestRecoverPartialJSON:
    """Tests for _recover_partial_json helper."""

    def test_empty_text(self) -> None:
        assert _recover_partial_json("") == []

    def test_no_json(self) -> None:
        assert _recover_partial_json("just plain text no braces") == []

    def test_single_complete_object(self) -> None:
        text = 'prefix {"start": 0.0, "end": 1.0, "text": "Hi"} suffix'
        result = _recover_partial_json(text)
        assert len(result) == 1
        assert result[0]["text"] == "Hi"

    def test_truncated_array(self) -> None:
        """Simulates truncated ASR output: array with 2 complete + 1 partial."""
        text = (
            '[{"start": 0.0, "end": 1.0, "speaker_id": 0, "text": "Hello"}, '
            '{"start": 1.5, "end": 3.0, "speaker_id": 0, "text": "World"}, '
            '{"start": 3.5, "end":'
        )
        result = _recover_partial_json(text)
        assert len(result) == 2
        assert result[0]["text"] == "Hello"
        assert result[1]["text"] == "World"

    def test_complete_array_still_works(self) -> None:
        """Complete JSON also works (recovers individual objects)."""
        text = '[{"start": 0, "end": 1, "text": "A"}, {"start": 2, "end": 3, "text": "B"}]'
        result = _recover_partial_json(text)
        assert len(result) == 2

    def test_normalise_falls_back_to_recovery(self) -> None:
        """_normalise_asr_segments uses partial recovery for truncated text."""

        @dataclass
        class _FakeResult:
            text: str
            segments: None = None

        truncated = (
            '[{"start": 0.0, "end": 1.0, "speaker_id": 0, "text": "Recovered"}, '
            '{"start": 2.0, "end":'
        )
        result = _FakeResult(text=truncated)
        segs = TranscribeStep._normalise_asr_segments(result)
        assert len(segs) == 1
        assert segs[0]["text"] == "Recovered"
