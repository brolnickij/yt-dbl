"""Step 4: Translate segments via Claude API (single-pass).

Sends all transcript segments to Claude Opus 4.6 in a single call.
The model translates, self-reflects, and returns the final adapted result
in one pass — leveraging the 1M token context window.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from yt_dbl.pipeline.base import PipelineStep
from yt_dbl.schemas import PipelineState, Segment, StepName
from yt_dbl.utils.logging import log_info

if TYPE_CHECKING:
    from pathlib import Path

TRANSLATIONS_FILE = "translations.json"
SUBTITLES_FILE = "subtitles.srt"

_SYSTEM_PROMPT = """\
You are a professional dubbing translator. The translated text will be \
read aloud by a TTS engine — optimize every translation for natural \
spoken delivery.

RULES:
1. Translate each segment naturally into {target_language}.
2. Preserve the speaker's register, tone, and style — formal/informal, \
technical jargon, humor, sarcasm, etc.
3. Keep translations CONCISE — they must fit the original segment's \
duration when spoken aloud. Shorter is always better than longer.
4. Segment durations: {duration_hint}. Each translation must be speakable \
within its time window.
5. Maintain consistent terminology and style across all segments.
6. Do NOT translate proper nouns, brand names, or technical terms commonly \
kept in the original language.
7. Write ALL numbers, dates, and numeric expressions as full words \
(e.g. "15%" → "fifteen percent", "2023" → "twenty twenty-three").
8. Expand abbreviations and units into spoken forms \
(e.g. "km/h" → "kilometers per hour"). For letter abbreviations, \
separate letters with spaces (e.g. "FBI" → "F B I").
9. Write for the EAR, not the eye: use short sentences, simple syntax, \
natural conversational flow. Avoid bookish or formal written style.
10. NEVER use characters that TTS cannot speak naturally: parentheses (), \
brackets [], slashes /, quotation marks. Rephrase in plain words instead.
11. SPLIT long sentences into SHORT ones (max 10-12 words each). \
TTS produces the best pronunciation on short, simple sentences. \
Use periods instead of semicolons or complex conjunctions.
12. For Russian: ALWAYS use the letter «ё» where it belongs \
(«всё» not «все» when meaning "everything", «ещё» not «еще», \
«её» not «ее» when meaning "her/hers", etc.). \
This helps TTS place stress correctly.
OUTPUT FORMAT — return ONLY a raw JSON array, no markdown, no commentary:
[
  {{"id": 0, "translated_text": "Translated text here"}},
  {{"id": 1, "translated_text": "Another segment translation"}}
]
"""


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
        raise TypeError("Expected a JSON array from Claude")

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
    description = "Translate via Claude Opus 4.6 (single-pass)"

    # ── validation ──────────────────────────────────────────────────────────

    def validate_inputs(self, state: PipelineState) -> None:
        if not state.segments:
            raise ValueError("No segments to translate")
        if not self.settings.anthropic_api_key:
            raise ValueError("Anthropic API key required — set YT_DBL_ANTHROPIC_API_KEY")

    # ── public API ──────────────────────────────────────────────────────────

    def run(self, state: PipelineState) -> PipelineState:
        translations_path = self.step_dir / TRANSLATIONS_FILE
        srt_path = self.step_dir / SUBTITLES_FILE

        # Idempotency: reuse existing result
        if translations_path.exists():
            log_info("Found existing translations — loading from cache")
            return self._load_cached(state, translations_path, srt_path)

        # Call Claude API
        translations = self._translate(state.segments, state.target_language)
        log_info(f"Translated {len(translations)}/{len(state.segments)} segments")

        # Apply translations to segments
        missing: list[int] = []
        for seg in state.segments:
            if seg.id in translations:
                seg.translated_text = translations[seg.id]
            else:
                missing.append(seg.id)

        if missing:
            log_info(f"Warning: {len(missing)} segments have no translation: {missing}")

        # Persist
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

    def _translate(
        self,
        segments: list[Segment],
        target_language: str,
    ) -> dict[int, str]:
        """Call Claude API with all segments and return translations."""
        from anthropic import Anthropic

        client = Anthropic(
            api_key=self.settings.anthropic_api_key,
            max_retries=3,
        )

        duration_hint = _build_duration_hint(segments)
        system_prompt = _SYSTEM_PROMPT.format(
            target_language=target_language,
            duration_hint=duration_hint,
        )
        user_message = _build_user_message(segments)

        log_info(f"Calling Claude ({self.settings.claude_model}) with {len(segments)} segments...")

        response = client.messages.create(
            model=self.settings.claude_model,
            max_tokens=16384,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )

        first_block = response.content[0]
        if not hasattr(first_block, "text"):  # pragma: no cover
            msg = f"Unexpected content block type: {type(first_block)}"
            raise TypeError(msg)
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
