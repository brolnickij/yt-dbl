"""Step 3: Transcribe speech + diarize speakers via mlx-audio.

Uses a two-model pipeline running entirely on Apple Silicon MLX Metal:
  1. VibeVoice-ASR (9B) — ASR + speaker diarization + segment timestamps
  2. Qwen3-ForcedAligner (0.6B) — word-level forced alignment
"""

from __future__ import annotations

import contextlib
import gc
import json
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from yt_dbl.pipeline.base import PipelineStep
from yt_dbl.schemas import PipelineState, Segment, Speaker, StepName, Word
from yt_dbl.utils.logging import console, create_progress, log_info, suppress_library_noise

if TYPE_CHECKING:
    from pathlib import Path

# Language code → full name for ForcedAligner API
_ALIGNER_LANGUAGE_MAP: dict[str, str] = {
    "en": "English",
    "ru": "Russian",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "hi": "Hindi",
    "tr": "Turkish",
    "nl": "Dutch",
    "pl": "Polish",
    "uk": "Ukrainian",
    "cs": "Czech",
    "sv": "Swedish",
    "vi": "Vietnamese",
    "th": "Thai",
    "id": "Indonesian",
    "ms": "Malay",
    "fi": "Finnish",
    "da": "Danish",
    "no": "Norwegian",
    "ro": "Romanian",
    "hu": "Hungarian",
    "el": "Greek",
    "he": "Hebrew",
    "bg": "Bulgarian",
    "hr": "Croatian",
    "sr": "Serbian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "et": "Estonian",
    "ka": "Georgian",
    "fa": "Persian",
    "ur": "Urdu",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "sw": "Swahili",
    "af": "Afrikaans",
    "ca": "Catalan",
    "eu": "Basque",
    "gl": "Galician",
    "cy": "Welsh",
    "is": "Icelandic",
}

SEGMENTS_FILE = "segments.json"

# Key-name variants produced by different VibeVoice versions
_START_KEYS = ("start", "start_time", "Start", "Start time")
_END_KEYS = ("end", "end_time", "End", "End time")
_SPEAKER_KEYS = ("speaker_id", "Speaker", "Speaker ID")
_TEXT_KEYS = ("text", "Content")


def _first_key(
    seg: dict[str, Any],
    keys: tuple[str, ...],
    *,
    cast: type = str,
) -> Any | None:
    """Return value of the first matching key, cast to *cast*, or None."""
    for key in keys:
        if key in seg:
            return cast(seg[key])
    return None


def _normalise_one_segment(seg: dict[str, Any]) -> dict[str, Any] | None:
    """Normalise a single raw ASR segment dict.  Returns *None* if incomplete."""
    start = _first_key(seg, _START_KEYS, cast=float)
    end = _first_key(seg, _END_KEYS, cast=float)
    text_val = _first_key(seg, _TEXT_KEYS)
    if start is None or end is None or text_val is None:
        return None

    speaker_id = _first_key(seg, _SPEAKER_KEYS, cast=int)
    return {
        "start": start,
        "end": end,
        "speaker_id": speaker_id if speaker_id is not None else 0,
        "text": str(text_val).strip(),
    }


# ── Reference segment scoring ───────────────────────────────────────────────

# Minimum duration (seconds) for a reference to be "good".
_MIN_REF_DURATION = 3.0
# Duration above which extra length stops helping.
_MAX_REF_DURATION = 8.0


def _reference_score(seg: Segment) -> float:
    """Score a segment for voice-reference quality.

    The score balances two signals:
      1. **Alignment confidence** - average ``Word.confidence`` across the
         segment.  Successfully aligned words have 1.0; segments where
         alignment failed fall back to 0.5 (see ``_align_segment``).
      2. **Duration sweet-spot** - duration is clamped to
         ``[0, _MAX_REF_DURATION]`` so very long (and potentially noisy)
         segments are not artificially preferred.

    A segment shorter than ``_MIN_REF_DURATION`` gets a small penalty
    (halved score) because very short clips give the TTS too little
    context for voice cloning.
    """
    avg_conf = sum(w.confidence for w in seg.words) / len(seg.words) if seg.words else 0.5

    dur = min(seg.duration, _MAX_REF_DURATION)
    score = avg_conf * dur

    if seg.duration < _MIN_REF_DURATION:
        score *= 0.5  # penalise very short segments

    return score


class TranscribeStep(PipelineStep):
    name = StepName.TRANSCRIBE
    description = "Transcribe + diarize speakers (VibeVoice-ASR + ForcedAligner)"

    # ── validation ──────────────────────────────────────────────────────────

    def validate_inputs(self, state: PipelineState) -> None:
        sep = state.get_step(StepName.SEPARATE)
        if "vocals" not in sep.outputs:
            raise ValueError("No vocals file from separation step")

    # ── public API ──────────────────────────────────────────────────────────

    def run(self, state: PipelineState) -> PipelineState:
        segments_path = self.step_dir / SEGMENTS_FILE

        # Idempotency: reuse existing result
        if segments_path.exists():
            log_info("Found existing transcription — loading from cache")
            return self._load_cached(state, segments_path)

        vocals_path = self._resolve_vocals(state)

        # Step 1: ASR + diarization
        raw_segments = self._run_asr(vocals_path)
        log_info(f"ASR produced {len(raw_segments)} segments")

        # Detect source language and store in state
        detected_lang = self._detect_language(raw_segments)
        state.source_language = detected_lang
        log_info(f"Detected source language: {detected_lang}")

        # Free ASR model before loading aligner (saves ~8 GB on constrained systems)
        asr_model = self.settings.transcription_asr_model
        if self.model_manager is not None and asr_model in self.model_manager.loaded_names:
            self.model_manager.unload(asr_model)

        # Step 2: word-level alignment
        segments = self._run_alignment(vocals_path, raw_segments, detected_lang)
        log_info(f"Alignment complete: {sum(len(s.words) for s in segments)} words")

        # Step 3: extract speakers
        speakers = self._extract_speakers(segments)
        log_info(f"Detected {len(speakers)} speakers")

        # Persist
        state.segments = segments
        state.speakers = speakers
        self._save(segments_path, segments, speakers)

        result = state.get_step(self.name)
        result.outputs = {"segments": SEGMENTS_FILE}

        return state

    # ── internals ───────────────────────────────────────────────────────────

    def _resolve_vocals(self, state: PipelineState) -> Path:
        sep_outputs = state.get_step(StepName.SEPARATE).outputs
        sep_dir = self.settings.step_dir(state.video_id, "02_separate")
        return sep_dir / sep_outputs["vocals"]

    # ── ASR (VibeVoice-ASR) ─────────────────────────────────────────────────

    def _run_asr(self, vocals_path: Path) -> list[dict[str, Any]]:
        """Run VibeVoice-ASR and return parsed segment dicts."""
        model_name = self.settings.transcription_asr_model

        if self.model_manager is not None:
            # Register loader if not yet registered
            if model_name not in self.model_manager.registered_names:
                self.model_manager.register(
                    model_name,
                    loader=lambda name=model_name: self._load_stt(name),
                )
            model = self.model_manager.get(model_name)
        else:
            model = self._load_stt(model_name)

        with console.status(
            "  [info]Running ASR + diarization (this may take several minutes)...[/info]",
            spinner="dots",
        ):
            result = model.generate(
                audio=str(vocals_path),
                max_tokens=self.settings.transcription_max_tokens,
                temperature=self.settings.transcription_temperature,
            )

        # VibeVoice key names vary across versions
        raw_segments = self._normalise_asr_segments(result)

        # If not managed, free manually
        if self.model_manager is None:
            del model
            gc.collect()

        return raw_segments

    @staticmethod
    def _load_stt(model_name: str) -> Any:
        """Load an STT model with noise suppression."""
        from mlx_audio.stt.utils import load as load_stt_model

        log_info(f"Loading model: {model_name}")
        with suppress_library_noise():
            return load_stt_model(model_name)

    @staticmethod
    def _normalise_asr_segments(result: Any) -> list[dict[str, Any]]:
        """Normalise VibeVoice output to a consistent format."""
        import re

        segments: list[dict[str, Any]] = []

        if hasattr(result, "segments") and result.segments:
            segments.extend(seg for seg in result.segments if isinstance(seg, dict))
        elif hasattr(result, "text"):
            # Fallback: try to parse JSON from raw text
            text = result.text
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                with contextlib.suppress(json.JSONDecodeError):
                    segments = json.loads(match.group())

        return [entry for seg in segments if (entry := _normalise_one_segment(seg)) is not None]

    # ── Forced alignment (Qwen3-ForcedAligner) ──────────────────────────────

    # Padding (seconds) added around each segment when slicing audio for
    # alignment.  Prevents cutting off speech at boundaries.
    _ALIGN_PAD_SEC: float = 0.5
    # ForcedAligner expects 16 kHz mono audio.
    _ALIGNER_SAMPLE_RATE: int = 16_000

    def _run_alignment(
        self,
        vocals_path: Path,
        raw_segments: list[dict[str, Any]],
        detected_lang: str = "en",
    ) -> list[Segment]:
        """Run Qwen3-ForcedAligner for word-level timestamps.

        Performance optimisation: the audio file is loaded into memory
        **once** and each segment is sliced from the in-memory array
        before being passed to the aligner.  This avoids re-reading the
        full file from disk and computing the mel-spectrogram over the
        entire recording for every segment.
        """
        aligner_name = self.settings.transcription_aligner_model

        if self.model_manager is not None:
            if aligner_name not in self.model_manager.registered_names:
                self.model_manager.register(
                    aligner_name,
                    loader=lambda name=aligner_name: self._load_stt(name),
                )
            aligner = self.model_manager.get(aligner_name)
        else:
            aligner = self._load_stt(aligner_name)

        lang_full = _ALIGNER_LANGUAGE_MAP.get(detected_lang, "English")

        # Load the whole audio file once (16 kHz mono mx.array)
        audio_array = self._load_audio_array(vocals_path)
        total_samples = audio_array.shape[0]

        segments: list[Segment] = []

        progress = create_progress()
        with progress:
            task = progress.add_task("  Aligning segments", total=len(raw_segments))
            for idx, seg in enumerate(raw_segments):
                text = seg["text"]
                if not text.strip():
                    progress.advance(task)
                    continue

                # Slice audio for this segment with padding
                pad_samples = int(self._ALIGN_PAD_SEC * self._ALIGNER_SAMPLE_RATE)
                start_sample = max(0, int(seg["start"] * self._ALIGNER_SAMPLE_RATE) - pad_samples)
                end_sample = min(
                    total_samples,
                    int(seg["end"] * self._ALIGNER_SAMPLE_RATE) + pad_samples,
                )
                audio_slice = audio_array[start_sample:end_sample]
                # Offset to add back to aligner timestamps (relative → absolute)
                time_offset = start_sample / self._ALIGNER_SAMPLE_RATE

                words = self._align_segment(
                    aligner,
                    audio_slice,
                    seg,
                    lang_full,
                    time_offset=time_offset,
                )

                segments.append(
                    Segment(
                        id=idx,
                        text=text,
                        start=seg["start"],
                        end=seg["end"],
                        speaker=f"SPEAKER_{seg['speaker_id']:02d}",
                        language=lang_full.lower()[:2],
                        words=words,
                    )
                )
                progress.advance(task)

        # If not managed, free manually
        if self.model_manager is None:
            del aligner
            gc.collect()

        return segments

    @staticmethod
    def _load_audio_array(vocals_path: Path) -> Any:
        """Load an audio file as a 16 kHz mono mx.array."""
        from mlx_audio.stt.utils import load_audio

        return load_audio(str(vocals_path), sr=16_000)

    @staticmethod
    def _align_segment(
        aligner: Any,
        audio: Any,
        seg: dict[str, Any],
        language: str,
        *,
        time_offset: float = 0.0,
    ) -> list[Word]:
        """Align a single segment and return Word list.

        Parameters
        ----------
        audio
            Pre-sliced mx.array (16 kHz mono) covering this segment.
        time_offset
            Seconds to add to every returned timestamp so that word
            times are in the coordinate system of the full recording.
        """
        text = seg["text"].strip()
        if not text:
            return []

        try:
            result = aligner.generate(
                audio=audio,
                text=text,
                language=language,
            )
        except Exception:
            # Fallback: create a single word spanning the whole segment
            return [Word(text=text, start=seg["start"], end=seg["end"], confidence=0.5)]

        words: list[Word] = []
        items = result.items if hasattr(result, "items") else list(result)
        for item in items:
            start = item.start_time if hasattr(item, "start_time") else item["start_time"]
            end = item.end_time if hasattr(item, "end_time") else item["end_time"]
            word_text = item.text if hasattr(item, "text") else item["text"]
            words.append(
                Word(
                    text=str(word_text),
                    start=float(start) + time_offset,
                    end=float(end) + time_offset,
                )
            )

        return words

    @staticmethod
    def _detect_language(raw_segments: list[dict[str, Any]]) -> str:
        """Best-effort language detection from segment text.

        Inspects Unicode script ranges to identify the dominant writing system.
        Falls back to 'en' for Latin-script languages (which VibeVoice handles
        well regardless).
        """
        all_text = " ".join(s.get("text", "") for s in raw_segments)

        # Count characters in various scripts
        cyrillic = sum(1 for c in all_text if "\u0400" <= c <= "\u04ff")
        latin = sum(1 for c in all_text if "a" <= c.lower() <= "z")
        arabic = sum(1 for c in all_text if "\u0600" <= c <= "\u06ff")
        devanagari = sum(1 for c in all_text if "\u0900" <= c <= "\u097f")
        thai = sum(1 for c in all_text if "\u0e00" <= c <= "\u0e7f")

        # Build script→language mapping for detection
        script_counts: dict[str, int] = {
            "ru": cyrillic,
            "ar": arabic,
            "hi": devanagari,
            "th": thai,
        }
        # Pick highest non-Latin script if it beats Latin
        best_script = max(script_counts, key=script_counts.get)  # type: ignore[arg-type]
        if script_counts[best_script] > latin:
            return best_script

        # Check Japanese kana BEFORE CJK (kanji are shared with Chinese)
        if any("\u3040" <= c <= "\u309f" or "\u30a0" <= c <= "\u30ff" for c in all_text):
            return "ja"
        if any("\uac00" <= c <= "\ud7af" for c in all_text):
            return "ko"
        if any("\u4e00" <= c <= "\u9fff" for c in all_text):
            return "zh"
        return "en"

    # ── Speaker extraction ──────────────────────────────────────────────────

    @staticmethod
    def _extract_speakers(segments: list[Segment]) -> list[Speaker]:
        """Build Speaker list from segment metadata."""
        durations: dict[str, float] = defaultdict(float)
        for seg in segments:
            durations[seg.speaker] += seg.duration

        speakers: list[Speaker] = []
        for speaker_id, total_dur in sorted(durations.items()):
            # Pick the cleanest segment as voice reference.
            # _reference_score balances alignment confidence and duration.
            best_seg = max(
                (s for s in segments if s.speaker == speaker_id),
                key=_reference_score,
            )
            speakers.append(
                Speaker(
                    id=speaker_id,
                    reference_start=best_seg.start,
                    reference_end=best_seg.end,
                    total_duration=round(total_dur, 3),
                )
            )

        return speakers

    # ── Persistence ─────────────────────────────────────────────────────────

    @staticmethod
    def _save(
        path: Path,
        segments: list[Segment],
        speakers: list[Speaker],
    ) -> None:
        data = {
            "segments": [s.model_dump() for s in segments],
            "speakers": [s.model_dump() for s in speakers],
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _load_cached(state: PipelineState, path: Path) -> PipelineState:
        data = json.loads(path.read_text(encoding="utf-8"))
        state.segments = [Segment.model_validate(s) for s in data["segments"]]
        state.speakers = [Speaker.model_validate(s) for s in data["speakers"]]

        result = state.get_step(StepName.TRANSCRIBE)
        result.outputs = {"segments": SEGMENTS_FILE}
        return state
