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


def _segments_fingerprint(segments: list[Segment]) -> str:
    """Compute a short hash fingerprint from segment IDs and source texts.

    Used to detect stale translation caches after re-transcription.
    When the upstream transcription step produces different segments,
    the fingerprint changes and cached translations are invalidated.
    """
    import hashlib

    payload = json.dumps(
        [(seg.id, seg.text) for seg in segments],
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


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

        # Idempotency: reuse existing result if segments haven't changed
        if translations_path.exists():
            cached = self._load_cached(state, translations_path, srt_path)
            if cached is not None:
                return cached
            # Segments changed — invalidate all caches before re-translating
            self._invalidate_caches()

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

        self._save(translations_path, translations, state.segments)
        _generate_srt(state.segments, srt_path)
        log_info(f"Generated subtitles: {SUBTITLES_FILE}")

        result = state.get_step(self.name)
        result.outputs = {
            "translations": TRANSLATIONS_FILE,
            "subtitles": SUBTITLES_FILE,
        }

        return state

    # ── Claude API ──────────────────────────────────────────────────────────

    # Maximum number of application-level retries when Claude returns
    # unparseable output (e.g. malformed JSON).  SDK-level HTTP retries
    # (429, 500, etc.) are handled separately by ``max_retries=3``.
    _MAX_PARSE_RETRIES: int = 2

    def _translate_all(
        self,
        segments: list[Segment],
        target_language: str,
        source_language: str = "auto-detected",
    ) -> dict[int, str]:
        """Translate all segments, batching automatically if needed.

        A single ``Anthropic`` client is created and reused across all
        batches.  Each batch result is cached to
        ``_translate_batch_NNN.json`` so that a resumed pipeline skips
        already-translated batches.  Cache files are cleaned up after a
        successful merge.
        """
        from anthropic import Anthropic

        client = Anthropic(
            api_key=self.settings.anthropic_api_key,
            max_retries=3,
        )

        batch_size = self.settings.translation_batch_size

        if len(segments) <= batch_size:
            return self._translate_batch(
                client,
                segments,
                target_language,
                source_language,
            )

        # Split into batches
        batches = [segments[i : i + batch_size] for i in range(0, len(segments), batch_size)]
        n_batches = len(batches)
        log_info(
            f"Splitting {len(segments)} segments into {n_batches} batches "
            f"({batch_size} segments each)"
        )

        all_translations: dict[int, str] = {}

        # Phase 1: load cached batches, collect uncached indices
        uncached: list[int] = []
        for idx, batch in enumerate(batches):
            cache_path = self.step_dir / f"_translate_batch_{idx:03d}.json"
            if cache_path.exists():
                raw = json.loads(cache_path.read_text(encoding="utf-8"))
                batch_fp = _segments_fingerprint(batch)

                if isinstance(raw, dict) and raw.get("_fingerprint") == batch_fp:
                    batch_result = {
                        int(item["id"]): str(item["translated_text"]) for item in raw["items"]
                    }
                    all_translations.update(batch_result)
                    log_info(
                        f"Batch {idx + 1}/{n_batches}: loaded from cache "
                        f"({len(batch_result)} translations)"
                    )
                    continue
                log_warning(f"Batch {idx + 1}/{n_batches}: cache is stale — re-translating")
                cache_path.unlink()
            uncached.append(idx)

        # Phase 2: translate uncached batches in parallel
        if uncached:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            def _do_batch(idx: int) -> tuple[int, dict[int, str]]:
                batch = batches[idx]
                log_info(f"Translating batch {idx + 1}/{n_batches} ({len(batch)} segments)")
                result = self._translate_batch(client, batch, target_language, source_language)
                cache_path = self.step_dir / f"_translate_batch_{idx:03d}.json"
                batch_data = {
                    "_fingerprint": _segments_fingerprint(batch),
                    "items": [{"id": k, "translated_text": v} for k, v in sorted(result.items())],
                }
                cache_path.write_text(
                    json.dumps(batch_data, ensure_ascii=False),
                    encoding="utf-8",
                )
                return idx, result

            max_workers = min(len(uncached), 4)  # cap concurrent API calls
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(_do_batch, idx): idx for idx in uncached}
                errors: list[tuple[int, Exception]] = []
                for future in as_completed(futures):
                    try:
                        idx, batch_result = future.result()
                        all_translations.update(batch_result)
                    except Exception as exc:
                        batch_idx = futures[future]
                        errors.append((batch_idx, exc))
                        log_warning(f"Batch {batch_idx + 1}/{n_batches} failed: {exc}")

                if errors:
                    failed = sorted(idx + 1 for idx, _ in errors)
                    raise TranslationError(
                        f"{len(errors)} translation batch(es) failed: {failed}"
                    ) from errors[0][1]

        # Clean up all batch cache files after successful merge
        for path in self.step_dir.glob("_translate_batch_*.json"):
            path.unlink()

        return all_translations

    def _translate_batch(
        self,
        client: Any,
        segments: list[Segment],
        target_language: str,
        source_language: str = "auto-detected",
    ) -> dict[int, str]:
        """Call Claude API with a batch of segments and return translations.

        Retries up to ``_MAX_PARSE_RETRIES`` times when Claude returns
        syntactically invalid output (malformed JSON, wrong structure).
        HTTP-level errors (429, 500) are retried by the SDK itself.
        """
        duration_hint = _build_duration_hint(segments)
        system_prompt = _load_system_prompt().format(
            target_language=target_language,
            source_language=source_language,
            duration_hint=duration_hint,
        )
        user_message = _build_user_message(segments)

        last_exc: Exception | None = None
        for attempt in range(1 + self._MAX_PARSE_RETRIES):
            if attempt > 0:
                log_warning(f"Retrying translation (attempt {attempt + 1})...")

            log_info(
                f"Calling Claude ({self.settings.claude_model}) with {len(segments)} segments..."
            )

            # Use streaming to avoid the Anthropic SDK 10-minute timeout
            # on long translation requests.  get_final_message() returns
            # the same Message object as messages.create().
            with client.messages.stream(
                model=self.settings.claude_model,
                max_tokens=self.settings.translation_max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            ) as stream:
                response = stream.get_final_message()

            first_block = response.content[0]
            if not hasattr(first_block, "text"):  # pragma: no cover
                msg = f"Unexpected content block type: {type(first_block)}"
                raise TranslationError(msg)
            response_text: str = first_block.text
            log_info(
                f"Claude response: {response.usage.input_tokens} in / "
                f"{response.usage.output_tokens} out tokens"
            )

            # Detect token-limit truncation — retrying won't help because
            # the same input will produce the same truncated output.
            if getattr(response, "stop_reason", None) == "max_tokens":
                raise TranslationError(
                    f"Claude output was truncated at "
                    f"{self.settings.translation_max_tokens} max_tokens "
                    f"(batch: {len(segments)} segments). "
                    "Increase YT_DBL_TRANSLATION_MAX_TOKENS or decrease "
                    "YT_DBL_TRANSLATION_BATCH_SIZE."
                )

            try:
                return _parse_translations(response_text)
            except (json.JSONDecodeError, TranslationError, KeyError, ValueError) as exc:
                last_exc = exc
                log_warning(
                    f"Failed to parse translation response: {exc} "
                    f"(preview: {response_text[:200]!r})"
                )

        # All retries exhausted
        raise TranslationError(
            f"Failed to get valid translation after {1 + self._MAX_PARSE_RETRIES} attempts"
        ) from last_exc

    # ── Persistence ─────────────────────────────────────────────────────────

    def _invalidate_caches(self) -> None:
        """Remove stale translation cache files from step directory."""
        for path in self.step_dir.glob("_translate_batch_*.json"):
            path.unlink()
        for name in (TRANSLATIONS_FILE, SUBTITLES_FILE):
            path = self.step_dir / name
            if path.exists():
                path.unlink()

    @staticmethod
    def _save(path: Path, translations: dict[int, str], segments: list[Segment]) -> None:
        """Save translations as JSON with a segment fingerprint for cache validation."""
        data = {
            "_fingerprint": _segments_fingerprint(segments),
            "items": [{"id": k, "translated_text": v} for k, v in sorted(translations.items())],
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _load_cached(
        state: PipelineState,
        translations_path: Path,
        srt_path: Path,
    ) -> PipelineState | None:
        """Load cached translations if the segment fingerprint still matches.

        Returns ``None`` when the cache is stale (segments changed since
        the translations were generated) or uses an old format without
        a fingerprint.
        """
        raw = json.loads(translations_path.read_text(encoding="utf-8"))

        # Old format (plain list) has no fingerprint — treat as stale
        if isinstance(raw, list):
            log_warning("Translations cache has no fingerprint — invalidating")
            return None

        current_fp = _segments_fingerprint(state.segments)
        if raw.get("_fingerprint") != current_fp:
            log_warning("Segments changed since last translation — invalidating cache")
            return None

        log_info("Found existing translations — loading from cache")
        items: list[dict[str, Any]] = raw["items"]
        translations = {int(item["id"]): str(item["translated_text"]) for item in items}

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
