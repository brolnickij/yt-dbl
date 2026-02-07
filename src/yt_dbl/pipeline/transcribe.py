"""Step 3: Transcribe speech + diarize speakers via mlx-audio.

Uses a two-model pipeline running entirely on Apple Silicon MLX Metal:
  1. VibeVoice-ASR (9B) — ASR + speaker diarization + segment timestamps
  2. Qwen3-ForcedAligner (0.6B) — word-level forced alignment
"""

from __future__ import annotations

import contextlib
import gc
import json
import re as _re_module
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from yt_dbl.pipeline.base import PipelineStep, StepValidationError, TranscriptionError
from yt_dbl.schemas import PipelineState, Segment, Speaker, StepName, Word
from yt_dbl.utils.languages import ALIGNER_LANGUAGE_MAP
from yt_dbl.utils.logging import (
    console,
    create_progress,
    log_info,
    log_warning,
    suppress_library_noise,
)

if TYPE_CHECKING:
    from pathlib import Path

    from yt_dbl.models.protocols import AlignerModel, STTModel

SEGMENTS_FILE = "segments.json"

# Key-name variants produced by different VibeVoice versions
_START_KEYS = ("start", "start_time", "Start", "Start time")
_END_KEYS = ("end", "end_time", "End", "End time")
_SPEAKER_KEYS = ("speaker_id", "Speaker", "Speaker ID")
_TEXT_KEYS = ("text", "Content")

# Threshold for warning about potential truncation.  VibeVoice-ASR
# defaults to max_tokens=8192 internally; hitting that limit with
# no segments strongly suggests the output was truncated.
_TRUNCATION_TOKEN_THRESHOLD = 8192

# ── Timestamp parsing ──────────────────────────────────────────────────────

# Pattern:  "HH:MM:SS", "MM:SS", "H:MM:SS", "M:SS"
_TS_RE = _re_module.compile(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?$")


def _parse_timestamp(value: Any) -> float | None:
    """Parse a timestamp value to seconds (float).

    Handles:
      - Numeric types (int / float) — returned as-is.
      - String floats like ``"12.5"`` — parsed directly.
      - ``"MM:SS"`` / ``"HH:MM:SS"`` strings — converted to seconds.

    Returns *None* when the value cannot be interpreted.
    """
    if isinstance(value, (int, float)):
        return float(value)

    if not isinstance(value, str):
        return None

    # Try plain float first ("12.5", "0", "123.456")
    try:
        return float(value)
    except ValueError:
        pass

    # Try MM:SS / HH:MM:SS
    m = _TS_RE.match(value.strip())
    if m:
        parts = m.groups()  # (first, second, third_or_None)
        if parts[2] is not None:
            # Format: HH:MM:SS
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        # Format: MM:SS
        return int(parts[0]) * 60 + int(parts[1])

    return None


def _first_key(
    seg: dict[str, Any],
    keys: tuple[str, ...],
    *,
    cast: type = str,
) -> Any | None:
    """Return value of the first matching key, cast to *cast*, or None.

    When *cast* is ``float``, delegates to :func:`_parse_timestamp` so
    that ``"MM:SS"`` / ``"HH:MM:SS"`` strings are handled correctly.
    """
    for key in keys:
        if key in seg:
            if cast is float:
                return _parse_timestamp(seg[key])
            try:
                return cast(seg[key])
            except (ValueError, TypeError):
                return None
    return None


def _normalise_one_segment(seg: dict[str, Any]) -> dict[str, Any] | None:
    """Normalise a single raw ASR segment dict.  Returns *None* if incomplete."""
    start = _first_key(seg, _START_KEYS, cast=float)
    end = _first_key(seg, _END_KEYS, cast=float)
    text_val = _first_key(seg, _TEXT_KEYS)
    if start is None or end is None or text_val is None:
        return None
    if end <= start:
        return None

    speaker_id = _first_key(seg, _SPEAKER_KEYS, cast=int)
    return {
        "start": start,
        "end": end,
        "speaker_id": speaker_id if speaker_id is not None else 0,
        "text": str(text_val).strip(),
    }


def _recover_partial_json(text: str) -> list[dict[str, Any]]:
    """Extract complete JSON objects from possibly-truncated text.

    When the ASR model output is cut short (e.g. by a ``max_tokens``
    limit), the overall ``[...]`` array is syntactically broken, but
    individual ``{...}`` objects that were fully emitted are still valid.
    This function finds all balanced ``{...}`` substrings and attempts
    ``json.loads`` on each one.
    """
    results: list[dict[str, Any]] = []
    i = 0
    while i < len(text):
        if text[i] == "{":
            depth = 0
            j = i
            while j < len(text):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[i : j + 1]
                        try:
                            obj = json.loads(candidate)
                            if isinstance(obj, dict):
                                results.append(obj)
                        except json.JSONDecodeError:
                            pass
                        i = j + 1
                        break
                j += 1
            else:
                # Unbalanced brace — rest of text is truncated
                break
        else:
            i += 1
    return results


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
            raise StepValidationError("No vocals file from separation step")

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

        state.segments = segments
        state.speakers = speakers
        self._save(segments_path, segments, speakers, source_language=detected_lang)

        result = state.get_step(self.name)
        result.outputs = {"segments": SEGMENTS_FILE}

        return state

    # ── internals ───────────────────────────────────────────────────────────

    def _resolve_vocals(self, state: PipelineState) -> Path:
        return self.resolve_step_file(state, StepName.SEPARATE, "vocals")

    # ── ASR (VibeVoice-ASR) ─────────────────────────────────────────────────

    def _run_asr(self, vocals_path: Path) -> list[dict[str, Any]]:
        """Run VibeVoice-ASR and return parsed segment dicts.

        If audio exceeds ``transcription_max_chunk_minutes`` the file is
        split into overlapping chunks, each processed independently, and
        results are merged with speaker-ID reconciliation across chunks.
        """
        model_name = self.settings.transcription_asr_model
        model = self._get_or_load_model(
            model_name,
            loader=lambda: self._load_stt(model_name),
        )

        try:
            duration_sec = self._get_audio_duration_sec(vocals_path)
            max_chunk_sec = self.settings.transcription_max_chunk_minutes * 60

            if duration_sec <= max_chunk_sec:
                raw_segments = self._run_asr_single(model, vocals_path)
            else:
                overlap_sec = self.settings.transcription_chunk_overlap_minutes * 60
                raw_segments = self._run_asr_chunked(
                    model,
                    vocals_path,
                    duration_sec,
                    max_chunk_sec,
                    overlap_sec,
                )
        except TranscriptionError:
            raise
        except Exception as exc:
            raise TranscriptionError(f"ASR model failed: {exc}") from exc
        finally:
            # If not managed, free manually
            if self.model_manager is None:
                del model
                gc.collect()

        return raw_segments

    @staticmethod
    def _load_stt(model_name: str) -> Any:
        """Load an STT/aligner model with noise suppression.

        Returns an opaque model object that satisfies either
        :class:`~yt_dbl.models.protocols.STTModel` or
        :class:`~yt_dbl.models.protocols.AlignerModel` depending on
        the weights loaded.
        """
        from mlx_audio.stt.utils import load as load_stt_model

        log_info(f"Loading model: {model_name}")
        with suppress_library_noise():
            return load_stt_model(model_name)

    # ── Single-pass & chunked ASR helpers ───────────────────────────────────

    def _run_asr_single(self, model: STTModel, audio_path: Path) -> list[dict[str, Any]]:
        """Run ASR on a single audio file (must fit within model limits).

        ``max_tokens`` is set to ``-1`` (unlimited) so that the model
        generates until EOS.  ``mlx_lm.generate_step`` treats ``-1`` as
        infinite because the internal counter never equals ``-1``.
        """
        with console.status(
            "  [info]Running ASR + diarization (this may take several minutes)...[/info]",
            spinner="dots",
        ):
            result = model.generate(
                audio=str(audio_path),
                max_tokens=-1,
                temperature=self.settings.transcription_temperature,
            )

        # Warn when the model may have been truncated (generation_tokens ≈ a
        # suspiciously round power-of-two limit set by the library default).
        gen_tokens = getattr(result, "generation_tokens", 0)
        if (
            gen_tokens
            and gen_tokens >= _TRUNCATION_TOKEN_THRESHOLD
            and not getattr(result, "segments", None)
        ):
            log_warning(
                f"ASR generated {gen_tokens} tokens but produced no segments — "
                "output may have been truncated"
            )

        return self._normalise_asr_segments(result)

    def _run_asr_chunked(
        self,
        model: STTModel,
        vocals_path: Path,
        duration_sec: float,
        max_chunk_sec: float,
        overlap_sec: float,
    ) -> list[dict[str, Any]]:
        """Split long audio into overlapping chunks, run ASR, merge results.

        After each chunk is transcribed its segments are persisted to
        ``_asr_chunk_NNN_segments.json`` inside ``step_dir``.  If the
        pipeline is interrupted and re-run, already-persisted chunks are
        loaded from cache instead of re-transcribing.
        """
        import soundfile as sf

        info = sf.info(str(vocals_path))
        sr = info.samplerate

        boundaries = self._compute_chunk_boundaries(duration_sec, max_chunk_sec, overlap_sec)
        n_chunks = len(boundaries)
        dur_min = duration_sec / 60
        log_info(
            f"Audio is {dur_min:.1f} min - splitting into {n_chunks} chunks "
            f"({max_chunk_sec / 60:.0f} min each, {overlap_sec / 60:.0f} min overlap)"
        )

        processed: list[tuple[float, float, list[dict[str, Any]]]] = []
        global_max_spk = -1

        progress = create_progress()
        with progress:
            task = progress.add_task("  ASR chunks", total=n_chunks)

            for i, (start_sec, end_sec) in enumerate(boundaries):
                raw = self._transcribe_single_chunk(
                    model,
                    vocals_path,
                    sr,
                    i,
                    n_chunks,
                    start_sec,
                    end_sec,
                )

                # Reconcile speaker IDs with previous chunk
                if i == 0:
                    global_max_spk = max(
                        (s["speaker_id"] for s in raw),
                        default=0,
                    )
                else:
                    _prev_start, prev_end = boundaries[i - 1]
                    mapping, global_max_spk = self._reconcile_chunk_speakers(
                        prev_segments=processed[i - 1][2],
                        curr_segments=raw,
                        overlap_start=start_sec,
                        overlap_end=prev_end,
                        global_max_speaker=global_max_spk,
                    )
                    for seg in raw:
                        seg["speaker_id"] = mapping.get(
                            seg["speaker_id"],
                            seg["speaker_id"],
                        )

                processed.append((start_sec, end_sec, raw))
                progress.advance(task)

        # Clean up chunk cache files after successful merge
        merged = self._merge_chunk_segments(processed, overlap_sec)
        for i in range(n_chunks):
            (self.step_dir / f"_asr_chunk_{i:03d}_segments.json").unlink(missing_ok=True)

        return merged

    @staticmethod
    def _compute_chunk_boundaries(
        duration_sec: float,
        max_chunk_sec: float,
        overlap_sec: float,
    ) -> list[tuple[float, float]]:
        """Return ``(start, end)`` pairs for overlapping audio chunks.

        Raises
        ------
        TranscriptionError
            If ``overlap_sec >= max_chunk_sec`` (would cause zero/negative
            step and an infinite loop).
        """
        if overlap_sec >= max_chunk_sec:
            msg = (
                f"Chunk overlap ({overlap_sec}s) must be less than "
                f"chunk duration ({max_chunk_sec}s)"
            )
            raise TranscriptionError(msg)

        boundaries: list[tuple[float, float]] = []
        chunk_start = 0.0
        step_sec = max_chunk_sec - overlap_sec
        while chunk_start < duration_sec:
            chunk_end = min(chunk_start + max_chunk_sec, duration_sec)
            boundaries.append((chunk_start, chunk_end))
            if chunk_end >= duration_sec:
                break
            chunk_start += step_sec
        return boundaries

    def _transcribe_single_chunk(
        self,
        model: STTModel,
        vocals_path: Path,
        sr: int,
        chunk_idx: int,
        n_chunks: int,
        start_sec: float,
        end_sec: float,
    ) -> list[dict[str, Any]]:
        """Transcribe one audio chunk, with caching for resume."""
        import soundfile as sf

        chunk_cache_path = self.step_dir / f"_asr_chunk_{chunk_idx:03d}_segments.json"

        if chunk_cache_path.exists():
            raw: list[dict[str, Any]] = json.loads(
                chunk_cache_path.read_text(encoding="utf-8"),
            )
            log_info(
                f"Chunk {chunk_idx + 1}/{n_chunks}: loaded from cache "
                f"({start_sec / 60:.1f}-{end_sec / 60:.1f} min, "
                f"{len(raw)} segments)"
            )
            return raw

        start_frame = int(start_sec * sr)
        num_frames = int((end_sec - start_sec) * sr)

        with sf.SoundFile(str(vocals_path)) as f:
            f.seek(start_frame)
            audio_data = f.read(num_frames)

        chunk_audio_path = self.step_dir / f"_asr_chunk_{chunk_idx:03d}.wav"
        sf.write(str(chunk_audio_path), audio_data, sr)
        del audio_data  # Free chunk buffer (~345 MB at 48 kHz × 30 min)

        try:
            raw = self._run_asr_single(model, chunk_audio_path)
        finally:
            chunk_audio_path.unlink(missing_ok=True)

        for seg in raw:
            seg["start"] += start_sec
            seg["end"] += start_sec

        chunk_cache_path.write_text(
            json.dumps(raw, ensure_ascii=False),
            encoding="utf-8",
        )

        log_info(
            f"Chunk {chunk_idx + 1}/{n_chunks}: {len(raw)} segments "
            f"({start_sec / 60:.1f}-{end_sec / 60:.1f} min)"
        )
        return raw

    @staticmethod
    def _get_audio_duration_sec(path: Path) -> float:
        """Return audio file duration in seconds."""
        import soundfile as sf

        return float(sf.info(str(path)).duration)

    @staticmethod
    def _merge_chunk_segments(
        chunk_results: list[tuple[float, float, list[dict[str, Any]]]],
        overlap_sec: float,
    ) -> list[dict[str, Any]]:
        """Merge segments from overlapping chunks, deduplicating overlap zones.

        Each chunk "owns" segments whose ``start`` falls within its zone.
        Zone boundaries are the midpoints of the overlap regions between
        adjacent chunks.
        """
        if not chunk_results:
            return []
        if len(chunk_results) == 1:
            return chunk_results[0][2]

        # Midpoints between adjacent overlap zones
        midpoints: list[float] = []
        for i in range(len(chunk_results) - 1):
            next_start = chunk_results[i + 1][0]
            midpoints.append(next_start + overlap_sec / 2)

        merged: list[dict[str, Any]] = []
        for i, (_start, _end, segments) in enumerate(chunk_results):
            lo = midpoints[i - 1] if i > 0 else 0.0
            hi = midpoints[i] if i < len(midpoints) else float("inf")
            merged.extend(seg for seg in segments if lo <= seg["start"] < hi)

        return merged

    @staticmethod
    def _reconcile_chunk_speakers(
        prev_segments: list[dict[str, Any]],
        curr_segments: list[dict[str, Any]],
        overlap_start: float,
        overlap_end: float,
        global_max_speaker: int,
    ) -> tuple[dict[int, int], int]:
        """Map current-chunk speaker IDs to match the previous chunk.

        Builds a temporal-overlap matrix for speaker pairs in the shared
        zone and solves optimal bipartite assignment via the Hungarian
        algorithm (``scipy.optimize.linear_sum_assignment``).  This
        guarantees the globally optimal mapping (maximum total overlap),
        unlike the previous greedy approach which could be suboptimal when
        a speaker has comparable overlap with multiple candidates.

        Unmatched speakers get fresh global IDs.

        Returns ``(mapping, new_global_max)`` where *mapping* is
        ``{curr_local_id: global_id}``.
        """
        from scipy.optimize import linear_sum_assignment

        prev_in_zone = [
            s for s in prev_segments if s["end"] > overlap_start and s["start"] < overlap_end
        ]
        curr_in_zone = [
            s for s in curr_segments if s["end"] > overlap_start and s["start"] < overlap_end
        ]

        all_curr_ids = sorted({s["speaker_id"] for s in curr_segments})
        gmax = global_max_speaker

        if not prev_in_zone or not curr_in_zone:
            # No overlap data - assign fresh global IDs
            mapping: dict[int, int] = {}
            for cid in all_curr_ids:
                gmax += 1
                mapping[cid] = gmax
            return mapping, gmax

        prev_ids = sorted({s["speaker_id"] for s in prev_in_zone})
        curr_ids = sorted({s["speaker_id"] for s in curr_in_zone})

        # Build overlap matrix: rows = curr speakers, cols = prev speakers
        overlap_matrix = []
        for cid in curr_ids:
            c_ivs = [(s["start"], s["end"]) for s in curr_in_zone if s["speaker_id"] == cid]
            row = []
            for pid in prev_ids:
                p_ivs = [(s["start"], s["end"]) for s in prev_in_zone if s["speaker_id"] == pid]
                total = 0.0
                for cs, ce in c_ivs:
                    for ps, pe in p_ivs:
                        ov = min(ce, pe) - max(cs, ps)
                        if ov > 0:
                            total += ov
                row.append(total)
            overlap_matrix.append(row)

        # Hungarian algorithm (minimises cost, so negate overlap to maximise)
        cost = [[-v for v in row] for row in overlap_matrix]
        row_idx, col_idx = linear_sum_assignment(cost)

        mapping = {}
        for ri, ci in zip(row_idx, col_idx, strict=True):
            if overlap_matrix[ri][ci] > 0:
                mapping[curr_ids[ri]] = prev_ids[ci]

        # Unmatched current speakers get new global IDs
        for cid in all_curr_ids:
            if cid not in mapping:
                gmax += 1
                mapping[cid] = gmax

        return mapping, gmax

    @staticmethod
    def _normalise_asr_segments(result: Any) -> list[dict[str, Any]]:
        """Normalise VibeVoice output to a consistent format.

        Falls back to partial-JSON recovery when the structured
        ``result.segments`` list is empty, which can happen when the
        model output was truncated (e.g. by a ``max_tokens`` limit).
        """
        import re

        segments: list[dict[str, Any]] = []

        if hasattr(result, "segments") and result.segments:
            segments.extend(seg for seg in result.segments if isinstance(seg, dict))
        elif hasattr(result, "text") and result.text:
            text = result.text
            # Strategy 1: find a complete JSON array
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                with contextlib.suppress(json.JSONDecodeError):
                    segments = json.loads(match.group())

            # Strategy 2 (partial recovery): extract individual {...} objects
            # from truncated JSON.  This rescues segments that were fully
            # serialised before the output was cut off.
            if not segments:
                recovered = _recover_partial_json(text)
                if recovered:
                    segments = recovered
                    log_warning(
                        f"Recovered {len(segments)} segments from truncated output "
                        f"(text length: {len(text)})"
                    )

        if not segments and hasattr(result, "text") and result.text:
            log_warning(
                f"ASR returned 0 parseable segments (raw text length: {len(result.text)}, "
                f"preview: {result.text[:200]!r})"
            )

        normalised = [
            entry for seg in segments if (entry := _normalise_one_segment(seg)) is not None
        ]

        dropped = len(segments) - len(normalised)
        if dropped > 0:
            log_warning(f"Dropped {dropped}/{len(segments)} segments with missing fields")

        return normalised

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
        aligner = self._get_or_load_model(
            aligner_name,
            loader=lambda: self._load_stt(aligner_name),
        )

        lang_full = ALIGNER_LANGUAGE_MAP.get(detected_lang, "English")

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
        aligner: AlignerModel,
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
        source_language: str = "",
    ) -> None:
        data: dict[str, Any] = {
            "segments": [s.model_dump() for s in segments],
            "speakers": [s.model_dump() for s in speakers],
        }
        if source_language:
            data["source_language"] = source_language
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _load_cached(state: PipelineState, path: Path) -> PipelineState:
        data = json.loads(path.read_text(encoding="utf-8"))
        state.segments = [Segment.model_validate(s) for s in data["segments"]]
        state.speakers = [Speaker.model_validate(s) for s in data["speakers"]]

        # Restore detected language when available (added in later versions)
        if source_lang := data.get("source_language"):
            state.source_language = source_lang

        result = state.get_step(StepName.TRANSCRIBE)
        result.outputs = {"segments": SEGMENTS_FILE}
        return state
