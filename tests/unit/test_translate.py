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
    """Create a fake Anthropic messages.create() response."""
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
        step, _, _state = _make_step(tmp_path)
        translations = {0: "Привет", 1: "Мир", 2: "Отлично"}

        path = step.step_dir / TRANSLATIONS_FILE
        step._save(path, translations)

        data = json.loads(path.read_text(encoding="utf-8"))
        assert len(data) == 3
        assert data[0]["translated_text"] == "Привет"

    def test_load_cached(self, tmp_path: Path) -> None:
        step, _, state = _make_step(tmp_path)
        translations = {0: "Привет", 1: "Мир", 2: "Отлично"}

        trans_path = step.step_dir / TRANSLATIONS_FILE
        srt_path = step.step_dir / SUBTITLES_FILE
        step._save(trans_path, translations)

        state = step._load_cached(state, trans_path, srt_path)

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
        fake_response = _fake_claude_response(translations)

        with patch("anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = fake_response
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

        # Pre-create cached translations
        translations = {0: "Cached A", 1: "Cached B", 2: "Cached C"}
        step._save(step.step_dir / TRANSLATIONS_FILE, translations)

        # Run should load from cache, not call API
        state = step.run(state)

        assert state.segments[0].translated_text == "Cached A"
        assert state.segments[2].translated_text == "Cached C"

    def test_run_handles_partial_translations(self, tmp_path: Path) -> None:
        """If Claude only returns some translations, missing ones are logged."""
        step, _, state = _make_step(tmp_path)
        translations = {0: "Only first"}  # missing 1 and 2
        fake_response = _fake_claude_response(translations)

        with patch("anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = fake_response
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
            mock_client.messages.create.side_effect = ConnectionError("network down")
            mock_cls.return_value = mock_client

            with pytest.raises(ConnectionError, match="network down"):
                step.run(state)

    def test_invalid_json_from_claude_raises(self, tmp_path: Path) -> None:
        """If Claude returns non-JSON text, a JSONDecodeError propagates."""
        step, _, state = _make_step(tmp_path)
        content_block = MagicMock()
        content_block.text = "Sorry, I cannot translate this."
        response = MagicMock()
        response.content = [content_block]
        response.usage = MagicMock(input_tokens=100, output_tokens=50)

        with patch("anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = response
            mock_cls.return_value = mock_client

            with pytest.raises(json.JSONDecodeError):
                step.run(state)

    def test_non_array_response_raises_translation_error(self, tmp_path: Path) -> None:
        """If Claude returns valid JSON but not an array, TranslationError is raised."""
        step, _, state = _make_step(tmp_path)
        content_block = MagicMock()
        content_block.text = '{"id": 0, "translated_text": "test"}'
        response = MagicMock()
        response.content = [content_block]
        response.usage = MagicMock(input_tokens=100, output_tokens=50)

        with patch("anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = response
            mock_cls.return_value = mock_client

            with pytest.raises(TranslationError, match="JSON array"):
                step.run(state)


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
        fake_response = _fake_claude_response(translations)

        with patch("anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = fake_response
            mock_cls.return_value = mock_client

            state = step.run(state)

        # Single call for 3 segments (batch_size=300)
        assert mock_client.messages.create.call_count == 1
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

        def make_response_for_batch(
            *, model: str, max_tokens: int, system: str, messages: list[dict[str, str]]
        ) -> MagicMock:
            user_msg = messages[0]["content"]
            items = json.loads(user_msg)
            translations = {item["id"]: f"Translated {item['id']}" for item in items}
            return _fake_claude_response(translations)

        with patch("anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_client.messages.create.side_effect = make_response_for_batch
            mock_cls.return_value = mock_client

            state = step.run(state)

        # 25 segments / 10 per batch = 3 API calls
        assert mock_client.messages.create.call_count == 3
        # All 25 segments translated
        for seg in state.segments:
            assert seg.translated_text == f"Translated {seg.id}"
