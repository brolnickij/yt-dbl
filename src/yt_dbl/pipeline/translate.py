"""Step 4: Translate segments via Claude API.

Sends transcript segments to Claude Sonnet 4.5 for translation.
For short videos everything is sent in a single call; for long audio
(>300 segments) the pipeline batches requests automatically to stay
within the model's output token limit.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path as _Path
from typing import TYPE_CHECKING, Any

from yt_dbl.pipeline.base import PipelineStep, StepValidationError, TranslationError
from yt_dbl.schemas import PipelineState, Segment, StepName
from yt_dbl.utils.logging import log_info, log_warning

if TYPE_CHECKING:
    from pathlib import Path

TRANSLATIONS_FILE = "translations.json"
SUBTITLES_FILE = "subtitles.srt"

_PROMPT_PATH = _Path(__file__).with_name("translate_prompt.txt")


@lru_cache(maxsize=1)
def _load_system_prompt() -> str:
    """Load the translation system prompt template from disk (cached)."""
    return _PROMPT_PATH.read_text(encoding="utf-8").rstrip("\n")


def _build_user_message(segments: list[Segment]) -> str:
    """Build the user message containing all segments to translate."""
    items: list[dict[str, Any]] = [
        {
            "id": seg.id,
            "speaker": seg.speaker,
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "duration_sec": round(seg.duration, 2),
            "text": seg.text,
        }
        for seg in segments
    ]
    return json.dumps(items, indent=2, ensure_ascii=False)


def _build_duration_hint(segments: list[Segment]) -> str:
    """Create a human-readable hint about segment durations."""
    durations = [seg.duration for seg in segments]
    if not durations:
        return "unknown"
    avg = sum(durations) / len(durations)
    return f"{min(durations):.1f}-{max(durations):.1f}s (avg {avg:.1f}s)"


def _parse_translations(response_text: str) -> dict[int, str]:
    """Parse Claude's JSON response into a mapping of segment ID → translated text."""
    import re

    # Extract JSON array from response (Claude may wrap it in markdown)
    text = response_text.strip()
    if text.startswith("```"):
        # Strip markdown code fence
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)

    items = json.loads(text)
    if not isinstance(items, list):
        raise TranslationError("Expected a JSON array from Claude")

    result: dict[int, str] = {}
    for item in items:
        seg_id = int(item["id"])
        translated = str(item["translated_text"]).strip()
        if translated:
            result[seg_id] = translated

    return result


def _generate_srt(segments: list[Segment], path: Path) -> None:
    """Write an SRT subtitle file from translated segments."""
    lines: list[str] = []
    for idx, seg in enumerate(segments, start=1):
        text = seg.translated_text or seg.text
        start = _format_srt_time(seg.start)
        end = _format_srt_time(seg.end)
        lines.append(f"{idx}")
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def _format_srt_time(seconds: float) -> str:
    """Format seconds as SRT timestamp: HH:MM:SS,mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


class TranslateStep(PipelineStep):
    name = StepName.TRANSLATE
    description = "Translate via Claude API (auto-batched)"

    # ── validation ──────────────────────────────────────────────────────────

    def validate_inputs(self, state: PipelineState) -> None:
        if not state.segments:
            raise StepValidationError("No segments to translate")
        if not self.settings.anthropic_api_key:
            raise StepValidationError("Anthropic API key required — set YT_DBL_ANTHROPIC_API_KEY")

    # ── public API ──────────────────────────────────────────────────────────

    def run(self, state: PipelineState) -> PipelineState:
        translations_path = self.step_dir / TRANSLATIONS_FILE
        srt_path = self.step_dir / SUBTITLES_FILE

        # Idempotency: reuse existing result
        if translations_path.exists():
            log_info("Found existing translations — loading from cache")
            return self._load_cached(state, translations_path, srt_path)

        source_lang = state.source_language or "auto-detected"
        translations = self._translate_all(state.segments, state.target_language, source_lang)
        log_info(f"Translated {len(translations)}/{len(state.segments)} segments")

        # Apply translations to segments
        missing: list[int] = []
        for seg in state.segments:
            if seg.id in translations:
                seg.translated_text = translations[seg.id]
            else:
                missing.append(seg.id)

        if missing:
            log_warning(f"{len(missing)} segments have no translation: {missing}")

        self._save(translations_path, translations)
        _generate_srt(state.segments, srt_path)
        log_info(f"Generated subtitles: {SUBTITLES_FILE}")

        result = state.get_step(self.name)
        result.outputs = {
            "translations": TRANSLATIONS_FILE,
            "subtitles": SUBTITLES_FILE,
        }

        return state

    # ── Claude API ──────────────────────────────────────────────────────────

    def _translate_all(
        self,
        segments: list[Segment],
        target_language: str,
        source_language: str = "auto-detected",
    ) -> dict[int, str]:
        """Translate all segments, batching automatically if needed."""
        batch_size = self.settings.translation_batch_size

        if len(segments) <= batch_size:
            return self._translate_batch(segments, target_language, source_language)

        # Split into batches
        batches = [segments[i : i + batch_size] for i in range(0, len(segments), batch_size)]
        log_info(
            f"Splitting {len(segments)} segments into {len(batches)} batches "
            f"({batch_size} segments each)"
        )

        all_translations: dict[int, str] = {}
        for idx, batch in enumerate(batches):
            log_info(f"Translating batch {idx + 1}/{len(batches)} ({len(batch)} segments)")
            batch_result = self._translate_batch(batch, target_language, source_language)
            all_translations.update(batch_result)

        return all_translations

    def _translate_batch(
        self,
        segments: list[Segment],
        target_language: str,
        source_language: str = "auto-detected",
    ) -> dict[int, str]:
        """Call Claude API with a batch of segments and return translations."""
        from anthropic import Anthropic

        client = Anthropic(
            api_key=self.settings.anthropic_api_key,
            max_retries=3,
        )

        duration_hint = _build_duration_hint(segments)
        system_prompt = _load_system_prompt().format(
            target_language=target_language,
            source_language=source_language,
            duration_hint=duration_hint,
        )
        user_message = _build_user_message(segments)

        log_info(f"Calling Claude ({self.settings.claude_model}) with {len(segments)} segments...")

        response = client.messages.create(
            model=self.settings.claude_model,
            max_tokens=self.settings.translation_max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )

        first_block = response.content[0]
        if not hasattr(first_block, "text"):  # pragma: no cover
            msg = f"Unexpected content block type: {type(first_block)}"
            raise TranslationError(msg)
        response_text: str = first_block.text
        log_info(
            f"Claude response: {response.usage.input_tokens} in / "
            f"{response.usage.output_tokens} out tokens"
        )

        return _parse_translations(response_text)

    # ── Persistence ─────────────────────────────────────────────────────────

    @staticmethod
    def _save(path: Path, translations: dict[int, str]) -> None:
        """Save translations as JSON."""
        data = [{"id": k, "translated_text": v} for k, v in sorted(translations.items())]
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _load_cached(
        state: PipelineState,
        translations_path: Path,
        srt_path: Path,
    ) -> PipelineState:
        """Load cached translations from disk."""
        data = json.loads(translations_path.read_text(encoding="utf-8"))
        translations = {int(item["id"]): str(item["translated_text"]) for item in data}

        for seg in state.segments:
            if seg.id in translations:
                seg.translated_text = translations[seg.id]

        # Regenerate SRT if missing
        if not srt_path.exists():
            _generate_srt(state.segments, srt_path)

        result = state.get_step(StepName.TRANSLATE)
        result.outputs = {
            "translations": TRANSLATIONS_FILE,
            "subtitles": SUBTITLES_FILE,
        }
        return state
