"""Tests for yt_dbl.pipeline.translate — translation step (mocked Claude API)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from yt_dbl.config import Settings
from yt_dbl.pipeline.base import StepValidationError, TranslationError
from yt_dbl.pipeline.translate import (
    SUBTITLES_FILE,
    TRANSLATIONS_FILE,
    TranslateStep,
    _build_duration_hint,
    _build_user_message,
    _format_srt_time,
    _generate_srt,
    _load_system_prompt,
    _parse_translations,
    _segments_fingerprint,
)
from yt_dbl.schemas import STEP_DIRS, PipelineState, Segment, StepName, StepStatus, Word

if TYPE_CHECKING:
    from pathlib import Path


# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_segments() -> list[Segment]:
    """Create test segments for translation."""
    return [
        Segment(
            id=0,
            text="Hello, welcome to this video.",
            start=0.0,
            end=3.5,
            speaker="SPEAKER_00",
            language="en",
            words=[Word(text="Hello,", start=0.0, end=0.5)],
        ),
        Segment(
            id=1,
            text="Today we'll talk about something interesting.",
            start=4.0,
            end=8.0,
            speaker="SPEAKER_00",
            language="en",
        ),
        Segment(
            id=2,
            text="That sounds great!",
            start=9.0,
            end=11.0,
            speaker="SPEAKER_01",
            language="en",
        ),
    ]


def _make_step(tmp_path: Path) -> tuple[TranslateStep, Settings, PipelineState]:
    cfg = Settings(work_dir=tmp_path / "work", anthropic_api_key="sk-test-key")
    step_dir = cfg.step_dir("test123", STEP_DIRS[StepName.TRANSLATE])
    step = TranslateStep(settings=cfg, work_dir=step_dir)
    state = PipelineState(video_id="test123", url="https://example.com")

    # Prefill transcription
    state.segments = _make_segments()

    trans = state.get_step(StepName.TRANSCRIBE)
    trans.status = StepStatus.COMPLETED
    trans.outputs = {"segments": "segments.json"}

    return step, cfg, state


def _fake_claude_response(translations: dict[int, str]) -> MagicMock:
    """Create a fake Anthropic Message object (same shape as non-streaming)."""
    items = [{"id": k, "translated_text": v} for k, v in translations.items()]
    response_text = json.dumps(items, ensure_ascii=False)

    content_block = MagicMock()
    content_block.text = response_text

    usage = MagicMock()
    usage.input_tokens = 500
    usage.output_tokens = 200

    response = MagicMock()
    response.content = [content_block]
    response.usage = usage
    return response


def _fake_claude_stream(translations: dict[int, str]) -> MagicMock:
    """Create a fake streaming context manager wrapping a response.

    Mimics ``client.messages.stream(...)`` used as a context manager
    whose ``get_final_message()`` returns a standard ``Message``.
    """
    response = _fake_claude_response(translations)
    stream = MagicMock()
    stream.get_final_message.return_value = response
    stream.__enter__ = MagicMock(return_value=stream)
    stream.__exit__ = MagicMock(return_value=False)
    return stream


# ── Validation tests ────────────────────────────────────────────────────────


class TestTranslateStepValidation:
    def test_validate_missing_segments(self, tmp_path: Path) -> None:
        step, _, state = _make_step(tmp_path)
        state.segments = []
        with pytest.raises(StepValidationError, match="No segments to translate"):
            step.validate_inputs(state)

    def test_validate_missing_api_key(self, tmp_path: Path) -> None:
        cfg = Settings(
            work_dir=tmp_path / "work",
            anthropic_api_key="",
        )
        step_dir = cfg.step_dir("test123", STEP_DIRS[StepName.TRANSLATE])
        step = TranslateStep(settings=cfg, work_dir=step_dir)
        state = PipelineState(video_id="test123")
        state.segments = _make_segments()
        with pytest.raises(StepValidationError, match="Anthropic API key"):
            step.validate_inputs(state)

    def test_validate_ok(self, tmp_path: Path) -> None:
        step, _, state = _make_step(tmp_path)
        step.validate_inputs(state)  # should not raise


# ── Message building tests ──────────────────────────────────────────────────


class TestBuildMessage:
    def test_user_message_contains_all_segments(self) -> None:
        segments = _make_segments()
        msg = _build_user_message(segments)
        data = json.loads(msg)
        assert len(data) == 3
        assert data[0]["id"] == 0
        assert data[0]["text"] == "Hello, welcome to this video."
        assert data[1]["duration_sec"] == 4.0
        assert data[2]["speaker"] == "SPEAKER_01"

    def test_duration_hint_format(self) -> None:
        segments = _make_segments()
        hint = _build_duration_hint(segments)
        assert "2.0" in hint  # min
        assert "4.0" in hint  # max
        assert "avg" in hint

    def test_duration_hint_empty(self) -> None:
        assert _build_duration_hint([]) == "unknown"

    def test_system_prompt_loads_and_formats(self) -> None:
        """The external prompt template can be loaded and formatted."""
        template = _load_system_prompt()
        assert "{source_language}" in template
        assert "{target_language}" in template
        assert "{duration_hint}" in template

        rendered = template.format(
            source_language="en",
            target_language="ru",
            duration_hint="1.0-5.0s (avg 3.0s)",
        )
        assert "en" in rendered
        assert "ru" in rendered
        assert "1.0-5.0s" in rendered
        # No unresolved placeholders remain (format fields like {foo})
        assert "{source_language}" not in rendered
        assert "{target_language}" not in rendered
        assert "{duration_hint}" not in rendered


# ── Parse translations tests ────────────────────────────────────────────────


class TestParseTranslations:
    def test_parse_plain_json(self) -> None:
        text = '[{"id": 0, "translated_text": "Привет"}, {"id": 1, "translated_text": "Мир"}]'
        result = _parse_translations(text)
        assert result == {0: "Привет", 1: "Мир"}

    def test_parse_markdown_fenced(self) -> None:
        text = '```json\n[{"id": 0, "translated_text": "Привет"}]\n```'
        result = _parse_translations(text)
        assert result == {0: "Привет"}

    def test_parse_markdown_no_lang(self) -> None:
        text = '```\n[{"id": 0, "translated_text": "Привет"}]\n```'
        result = _parse_translations(text)
        assert result == {0: "Привет"}

    def test_parse_skips_empty(self) -> None:
        text = '[{"id": 0, "translated_text": "OK"}, {"id": 1, "translated_text": ""}]'
        result = _parse_translations(text)
        assert result == {0: "OK"}

    def test_parse_invalid_json_raises(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            _parse_translations("not json at all")

    def test_parse_non_array_raises(self) -> None:
        with pytest.raises(TranslationError, match="JSON array"):
            _parse_translations('{"id": 0, "translated_text": "test"}')


# ── SRT generation tests ────────────────────────────────────────────────────


class TestSRTGeneration:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (0.0, "00:00:00,000"),
            (3.5, "00:00:03,500"),
            (65.123, "00:01:05,123"),
            (3661.0, "01:01:01,000"),
        ],
        ids=["zero", "fractional", "over-minute", "over-hour"],
    )
    def test_format_srt_time(self, seconds: float, expected: str) -> None:
        assert _format_srt_time(seconds) == expected

    def test_generate_srt_uses_translated_text(self, tmp_path: Path) -> None:
        segments = _make_segments()
        segments[0].translated_text = "Привет, добро пожаловать."
        segments[1].translated_text = "Сегодня поговорим."
        segments[2].translated_text = "Отлично!"

        srt_path = tmp_path / "test.srt"
        _generate_srt(segments, srt_path)

        content = srt_path.read_text(encoding="utf-8")
        assert "Привет, добро пожаловать." in content
        assert "00:00:00,000 --> 00:00:03,500" in content
        assert "3\n00:00:09,000 --> 00:00:11,000\nОтлично!" in content

    def test_generate_srt_falls_back_to_original(self, tmp_path: Path) -> None:
        segments = [Segment(id=0, text="Original text", start=0.0, end=1.0)]
        srt_path = tmp_path / "test.srt"
        _generate_srt(segments, srt_path)

        content = srt_path.read_text(encoding="utf-8")
        assert "Original text" in content


# ── Persistence tests ───────────────────────────────────────────────────────


class TestPersistence:
    def test_save_and_load(self, tmp_path: Path) -> None:
        step, _, state = _make_step(tmp_path)
        translations = {0: "Привет", 1: "Мир", 2: "Отлично"}

        path = step.step_dir / TRANSLATIONS_FILE
        step._save(path, translations, state.segments)

        data = json.loads(path.read_text(encoding="utf-8"))
        assert "_fingerprint" in data
        assert len(data["items"]) == 3
        assert data["items"][0]["translated_text"] == "Привет"

    def test_load_cached(self, tmp_path: Path) -> None:
        step, _, state = _make_step(tmp_path)
        translations = {0: "Привет", 1: "Мир", 2: "Отлично"}

        trans_path = step.step_dir / TRANSLATIONS_FILE
        srt_path = step.step_dir / SUBTITLES_FILE
        step._save(trans_path, translations, state.segments)

        result = step._load_cached(state, trans_path, srt_path)

        assert result is not None
        assert state.segments[0].translated_text == "Привет"
        assert state.segments[2].translated_text == "Отлично"
        assert srt_path.exists()  # SRT regenerated


# ── Full run tests (mocked API) ────────────────────────────────────────────


class TestTranslateStepRun:
    def test_run_success(self, tmp_path: Path) -> None:
        step, _, state = _make_step(tmp_path)
        translations = {
            0: "Привет, добро пожаловать.",
            1: "Сегодня поговорим об интересном.",
            2: "Звучит отлично!",
        }
        fake_stream = _fake_claude_stream(translations)

        with patch("anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_client.messages.stream.return_value = fake_stream
            mock_cls.return_value = mock_client

            state = step.run(state)

        assert state.segments[0].translated_text == "Привет, добро пожаловать."
        assert state.segments[2].translated_text == "Звучит отлично!"
        assert (step.step_dir / TRANSLATIONS_FILE).exists()
        assert (step.step_dir / SUBTITLES_FILE).exists()

        result = state.get_step(StepName.TRANSLATE)
        assert result.outputs["translations"] == TRANSLATIONS_FILE
        assert result.outputs["subtitles"] == SUBTITLES_FILE

    def test_run_idempotent(self, tmp_path: Path) -> None:
        step, _, state = _make_step(tmp_path)

        # Pre-create cached translations with correct fingerprint
        translations = {0: "Cached A", 1: "Cached B", 2: "Cached C"}
        step._save(step.step_dir / TRANSLATIONS_FILE, translations, state.segments)

        # Run should load from cache, not call API
        state = step.run(state)

        assert state.segments[0].translated_text == "Cached A"
        assert state.segments[2].translated_text == "Cached C"

    def test_run_handles_partial_translations(self, tmp_path: Path) -> None:
        """If Claude only returns some translations, missing ones are logged."""
        step, _, state = _make_step(tmp_path)
        translations = {0: "Only first"}  # missing 1 and 2
        fake_stream = _fake_claude_stream(translations)

        with patch("anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_client.messages.stream.return_value = fake_stream
            mock_cls.return_value = mock_client

            state = step.run(state)

        assert state.segments[0].translated_text == "Only first"
        assert state.segments[1].translated_text == ""
        assert state.segments[2].translated_text == ""


# ── API error tests ─────────────────────────────────────────────────────────


class TestTranslateAPIErrors:
    def test_connection_error_propagates(self, tmp_path: Path) -> None:
        """Network failures bubble up to the caller."""
        step, _, state = _make_step(tmp_path)
        with patch("anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_client.messages.stream.side_effect = ConnectionError("network down")
            mock_cls.return_value = mock_client

            with pytest.raises(ConnectionError, match="network down"):
                step.run(state)

    def test_invalid_json_retries_then_raises(self, tmp_path: Path) -> None:
        """If Claude returns non-JSON text, retries are attempted before failing."""
        step, _, state = _make_step(tmp_path)
        content_block = MagicMock()
        content_block.text = "Sorry, I cannot translate this."
        response = MagicMock()
        response.content = [content_block]
        response.usage = MagicMock(input_tokens=100, output_tokens=50)

        stream = MagicMock()
        stream.get_final_message.return_value = response
        stream.__enter__ = MagicMock(return_value=stream)
        stream.__exit__ = MagicMock(return_value=False)

        with patch("anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_client.messages.stream.return_value = stream
            mock_cls.return_value = mock_client

            with pytest.raises(TranslationError, match="Failed to get valid translation"):
                step.run(state)

        # 1 initial + 2 retries = 3 total calls
        assert mock_client.messages.stream.call_count == 3

    def test_non_array_retries_then_raises(self, tmp_path: Path) -> None:
        """If Claude returns valid JSON but not an array, retries then raises."""
        step, _, state = _make_step(tmp_path)
        content_block = MagicMock()
        content_block.text = '{"id": 0, "translated_text": "test"}'
        response = MagicMock()
        response.content = [content_block]
        response.usage = MagicMock(input_tokens=100, output_tokens=50)

        stream = MagicMock()
        stream.get_final_message.return_value = response
        stream.__enter__ = MagicMock(return_value=stream)
        stream.__exit__ = MagicMock(return_value=False)

        with patch("anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_client.messages.stream.return_value = stream
            mock_cls.return_value = mock_client

            with pytest.raises(TranslationError, match="Failed to get valid translation"):
                step.run(state)

        assert mock_client.messages.stream.call_count == 3

    def test_parse_retry_succeeds_on_second_attempt(self, tmp_path: Path) -> None:
        """If first response is bad JSON but second is valid, translation succeeds."""
        step, _, state = _make_step(tmp_path)

        bad_block = MagicMock()
        bad_block.text = "not json"
        bad_response = MagicMock()
        bad_response.content = [bad_block]
        bad_response.usage = MagicMock(input_tokens=100, output_tokens=50)
        bad_stream = MagicMock()
        bad_stream.get_final_message.return_value = bad_response
        bad_stream.__enter__ = MagicMock(return_value=bad_stream)
        bad_stream.__exit__ = MagicMock(return_value=False)

        good_stream = _fake_claude_stream({0: "A", 1: "B", 2: "C"})

        with patch("anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_client.messages.stream.side_effect = [bad_stream, good_stream]
            mock_cls.return_value = mock_client

            state = step.run(state)

        assert mock_client.messages.stream.call_count == 2
        assert state.segments[0].translated_text == "A"

    def test_max_tokens_truncation_raises_immediately(self, tmp_path: Path) -> None:
        """If Claude hits max_tokens limit, TranslationError is raised without retries."""
        step, _, state = _make_step(tmp_path)

        content_block = MagicMock()
        content_block.text = '[{"id": 0, "translated_text": "Trun'  # truncated JSON
        response = MagicMock()
        response.content = [content_block]
        response.usage = MagicMock(input_tokens=500, output_tokens=32768)
        response.stop_reason = "max_tokens"

        stream = MagicMock()
        stream.get_final_message.return_value = response
        stream.__enter__ = MagicMock(return_value=stream)
        stream.__exit__ = MagicMock(return_value=False)

        with patch("anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_client.messages.stream.return_value = stream
            mock_cls.return_value = mock_client

            with pytest.raises(TranslationError, match="truncated"):
                step.run(state)

        # Only 1 API call — no retries for truncation
        assert mock_client.messages.stream.call_count == 1


# ── Config tests ────────────────────────────────────────────────────────────


class TestTranslationConfig:
    def test_default_claude_model(self) -> None:
        cfg = Settings()
        assert cfg.claude_model == "claude-sonnet-4-5"

    def test_custom_model_via_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("YT_DBL_CLAUDE_MODEL", "claude-sonnet-4-5")
        cfg = Settings()
        assert cfg.claude_model == "claude-sonnet-4-5"

    def test_default_translation_batch_size(self) -> None:
        cfg = Settings()
        assert cfg.translation_batch_size == 300

    def test_default_translation_max_tokens(self) -> None:
        cfg = Settings()
        assert cfg.translation_max_tokens == 32768

    def test_custom_batch_size_via_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("YT_DBL_TRANSLATION_BATCH_SIZE", "500")
        cfg = Settings()
        assert cfg.translation_batch_size == 500


# ── Batched translation tests ───────────────────────────────────────────────


class TestBatchedTranslation:
    def _make_many_segments(self, n: int) -> list[Segment]:
        """Create n dummy segments."""
        return [
            Segment(
                id=i,
                text=f"Segment number {i}.",
                start=float(i * 5),
                end=float(i * 5 + 4),
                speaker=f"SPEAKER_{i % 2:02d}",
                language="en",
            )
            for i in range(n)
        ]

    def test_small_input_single_batch(self, tmp_path: Path) -> None:
        """Fewer segments than batch_size -> single API call."""
        step, _, state = _make_step(tmp_path)
        translations = {0: "A", 1: "B", 2: "C"}
        fake_stream = _fake_claude_stream(translations)

        with patch("anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_client.messages.stream.return_value = fake_stream
            mock_cls.return_value = mock_client

            state = step.run(state)

        # Single call for 3 segments (batch_size=300)
        assert mock_client.messages.stream.call_count == 1
        assert state.segments[0].translated_text == "A"

    def test_large_input_multiple_batches(self, tmp_path: Path) -> None:
        """More segments than batch_size -> multiple API calls."""
        cfg = Settings(
            work_dir=tmp_path / "work",
            anthropic_api_key="sk-test-key",
            translation_batch_size=10,
        )
        step_dir = cfg.step_dir("test123", STEP_DIRS[StepName.TRANSLATE])
        step = TranslateStep(settings=cfg, work_dir=step_dir)
        state = PipelineState(video_id="test123", url="https://example.com")
        state.segments = self._make_many_segments(25)
        trans = state.get_step(StepName.TRANSCRIBE)
        trans.status = StepStatus.COMPLETED
        trans.outputs = {"segments": "segments.json"}

        def make_stream_for_batch(
            *, model: str, max_tokens: int, system: str, messages: list[dict[str, str]]
        ) -> MagicMock:
            user_msg = messages[0]["content"]
            items = json.loads(user_msg)
            translations = {item["id"]: f"Translated {item['id']}" for item in items}
            return _fake_claude_stream(translations)

        with patch("anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_client.messages.stream.side_effect = make_stream_for_batch
            mock_cls.return_value = mock_client

            state = step.run(state)

        # 25 segments / 10 per batch = 3 API calls
        assert mock_client.messages.stream.call_count == 3
        # All 25 segments translated
        for seg in state.segments:
            assert seg.translated_text == f"Translated {seg.id}"


# ── Batch caching / resume tests ────────────────────────────────────────────


class TestBatchCaching:
    """Tests for per-batch caching in _translate_all."""

    def _make_many_segments(self, n: int) -> list[Segment]:
        return [
            Segment(
                id=i,
                text=f"Segment {i}.",
                start=float(i * 5),
                end=float(i * 5 + 4),
                speaker="SPEAKER_00",
                language="en",
            )
            for i in range(n)
        ]

    def test_cached_batches_skip_api_call(self, tmp_path: Path) -> None:
        """Pre-existing batch cache files are loaded without calling the API."""
        cfg = Settings(
            work_dir=tmp_path / "work",
            anthropic_api_key="sk-test-key",
            translation_batch_size=10,
        )
        step_dir = cfg.step_dir("test123", STEP_DIRS[StepName.TRANSLATE])
        step = TranslateStep(settings=cfg, work_dir=step_dir)
        state = PipelineState(video_id="test123", url="https://example.com")
        state.segments = self._make_many_segments(25)
        trans = state.get_step(StepName.TRANSCRIBE)
        trans.status = StepStatus.COMPLETED
        trans.outputs = {"segments": "segments.json"}

        # Pre-create cache for batch 0 and 1 (ids 0-9, 10-19) with fingerprints
        for batch_idx, id_range in enumerate([(0, 10), (10, 20)]):
            batch_segments = state.segments[id_range[0] : id_range[1]]
            batch_data = {
                "_fingerprint": _segments_fingerprint(batch_segments),
                "items": [
                    {"id": i, "translated_text": f"Cached {i}"}
                    for i in range(id_range[0], id_range[1])
                ],
            }
            cache_path = step_dir / f"_translate_batch_{batch_idx:03d}.json"
            cache_path.write_text(json.dumps(batch_data), encoding="utf-8")

        # Only batch 2 (ids 20-24) needs API call
        def make_stream_for_batch(
            *, model: str, max_tokens: int, system: str, messages: list[dict[str, str]]
        ) -> MagicMock:
            user_msg = messages[0]["content"]
            items = json.loads(user_msg)
            translations = {item["id"]: f"Translated {item['id']}" for item in items}
            return _fake_claude_stream(translations)

        with patch("anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_client.messages.stream.side_effect = make_stream_for_batch
            mock_cls.return_value = mock_client

            state = step.run(state)

        # Only 1 API call (batch 2), not 3
        assert mock_client.messages.stream.call_count == 1
        # Cached batches loaded correctly
        assert state.segments[5].translated_text == "Cached 5"
        assert state.segments[15].translated_text == "Cached 15"
        # API batch translated correctly
        assert state.segments[22].translated_text == "Translated 22"

    def test_batch_caches_cleaned_after_success(self, tmp_path: Path) -> None:
        """Batch cache files are removed after successful merge."""
        cfg = Settings(
            work_dir=tmp_path / "work",
            anthropic_api_key="sk-test-key",
            translation_batch_size=10,
        )
        step_dir = cfg.step_dir("test123", STEP_DIRS[StepName.TRANSLATE])
        step = TranslateStep(settings=cfg, work_dir=step_dir)
        state = PipelineState(video_id="test123", url="https://example.com")
        state.segments = self._make_many_segments(25)
        trans = state.get_step(StepName.TRANSCRIBE)
        trans.status = StepStatus.COMPLETED
        trans.outputs = {"segments": "segments.json"}

        def make_stream_for_batch(
            *, model: str, max_tokens: int, system: str, messages: list[dict[str, str]]
        ) -> MagicMock:
            user_msg = messages[0]["content"]
            items = json.loads(user_msg)
            translations = {item["id"]: f"T{item['id']}" for item in items}
            return _fake_claude_stream(translations)

        with patch("anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_client.messages.stream.side_effect = make_stream_for_batch
            mock_cls.return_value = mock_client

            step.run(state)

        # Batch cache files should be cleaned up
        for i in range(3):
            assert not (step_dir / f"_translate_batch_{i:03d}.json").exists()
        # But final translations.json should exist
        assert (step_dir / TRANSLATIONS_FILE).exists()

    def test_single_batch_no_cache_files(self, tmp_path: Path) -> None:
        """Single-batch translation (<=batch_size) doesn't create cache files."""
        step, _, state = _make_step(tmp_path)
        fake_stream = _fake_claude_stream({0: "A", 1: "B", 2: "C"})

        with patch("anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_client.messages.stream.return_value = fake_stream
            mock_cls.return_value = mock_client

            step.run(state)

        # No batch cache files for single-batch run
        cache_files = list(step.step_dir.glob("_translate_batch_*.json"))
        assert cache_files == []


# ── Stale cache validation tests ────────────────────────────────────────────


class TestStaleCacheValidation:
    """Tests for fingerprint-based cache invalidation."""

    def test_segments_fingerprint_deterministic(self) -> None:
        """Same segments produce the same fingerprint."""
        segments = _make_segments()
        assert _segments_fingerprint(segments) == _segments_fingerprint(segments)

    def test_segments_fingerprint_changes_on_text(self) -> None:
        """Changing segment text changes the fingerprint."""
        seg_a = _make_segments()
        seg_b = _make_segments()
        seg_b[0].text = "Completely different text."
        assert _segments_fingerprint(seg_a) != _segments_fingerprint(seg_b)

    def test_segments_fingerprint_changes_on_id(self) -> None:
        """Changing segment IDs changes the fingerprint."""
        seg_a = _make_segments()
        seg_b = _make_segments()
        seg_b[0].id = 999
        assert _segments_fingerprint(seg_a) != _segments_fingerprint(seg_b)

    def test_stale_cache_invalidated_on_segment_change(self, tmp_path: Path) -> None:
        """Cached translations are discarded when segments change."""
        step, _, state = _make_step(tmp_path)
        translations = {0: "Old A", 1: "Old B", 2: "Old C"}
        step._save(step.step_dir / TRANSLATIONS_FILE, translations, state.segments)

        # Simulate re-transcription producing different text
        state.segments[0].text = "Completely new transcription."
        fake_stream = _fake_claude_stream({0: "New A", 1: "New B", 2: "New C"})

        with patch("anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_client.messages.stream.return_value = fake_stream
            mock_cls.return_value = mock_client

            state = step.run(state)

        # Old cache was invalidated, new translations applied
        assert mock_client.messages.stream.call_count == 1
        assert state.segments[0].translated_text == "New A"

    def test_old_format_cache_invalidated(self, tmp_path: Path) -> None:
        """Old-format translations.json (plain list) is treated as stale."""
        step, _, state = _make_step(tmp_path)

        # Write old-format cache (plain list, no fingerprint)
        old_data = [
            {"id": 0, "translated_text": "Old A"},
            {"id": 1, "translated_text": "Old B"},
            {"id": 2, "translated_text": "Old C"},
        ]
        (step.step_dir / TRANSLATIONS_FILE).write_text(json.dumps(old_data), encoding="utf-8")
        fake_stream = _fake_claude_stream({0: "New A", 1: "New B", 2: "New C"})

        with patch("anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_client.messages.stream.return_value = fake_stream
            mock_cls.return_value = mock_client

            state = step.run(state)

        assert mock_client.messages.stream.call_count == 1
        assert state.segments[0].translated_text == "New A"

    def test_stale_batch_cache_re_translated(self, tmp_path: Path) -> None:
        """Batch caches with wrong fingerprint are re-translated."""
        cfg = Settings(
            work_dir=tmp_path / "work",
            anthropic_api_key="sk-test-key",
            translation_batch_size=10,
        )
        step_dir = cfg.step_dir("test123", STEP_DIRS[StepName.TRANSLATE])
        step = TranslateStep(settings=cfg, work_dir=step_dir)
        state = PipelineState(video_id="test123", url="https://example.com")
        state.segments = [
            Segment(
                id=i,
                text=f"Segment {i}.",
                start=float(i * 5),
                end=float(i * 5 + 4),
                speaker="SPEAKER_00",
                language="en",
            )
            for i in range(20)
        ]
        trans = state.get_step(StepName.TRANSCRIBE)
        trans.status = StepStatus.COMPLETED
        trans.outputs = {"segments": "segments.json"}

        # Pre-create STALE batch 0 cache (wrong fingerprint)
        stale_data = {
            "_fingerprint": "0000000000000000",
            "items": [{"id": i, "translated_text": f"Stale {i}"} for i in range(10)],
        }
        (step_dir / "_translate_batch_000.json").write_text(
            json.dumps(stale_data), encoding="utf-8"
        )

        def make_stream_for_batch(
            *, model: str, max_tokens: int, system: str, messages: list[dict[str, str]]
        ) -> MagicMock:
            user_msg = messages[0]["content"]
            items = json.loads(user_msg)
            translations = {item["id"]: f"Fresh {item['id']}" for item in items}
            return _fake_claude_stream(translations)

        with patch("anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_client.messages.stream.side_effect = make_stream_for_batch
            mock_cls.return_value = mock_client

            state = step.run(state)

        # Both batches needed API calls (batch 0 was stale)
        assert mock_client.messages.stream.call_count == 2
        assert state.segments[0].translated_text == "Fresh 0"
        assert state.segments[15].translated_text == "Fresh 15"

    def test_invalidate_caches_cleans_all_files(self, tmp_path: Path) -> None:
        """_invalidate_caches removes translations, subtitles, and batch caches."""
        step, _, state = _make_step(tmp_path)

        # Create various cache files
        step._save(step.step_dir / TRANSLATIONS_FILE, {0: "A"}, state.segments)
        (step.step_dir / SUBTITLES_FILE).write_text("1\n00:00:00,000 --> 00:00:01,000\nA\n")
        for i in range(5):
            (step.step_dir / f"_translate_batch_{i:03d}.json").write_text("{}")

        step._invalidate_caches()

        assert not (step.step_dir / TRANSLATIONS_FILE).exists()
        assert not (step.step_dir / SUBTITLES_FILE).exists()
        assert list(step.step_dir.glob("_translate_batch_*.json")) == []
