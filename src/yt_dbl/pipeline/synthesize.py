"""Step 5: Synthesize translated speech with voice cloning.

For each speaker, extracts a voice reference from vocals.wav, then uses
Qwen3-TTS via mlx-audio to synthesize each translated segment with the
cloned voice.  Over-long segments are sped up via ffmpeg (atempo filter).
"""

from __future__ import annotations

import gc
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

from yt_dbl.pipeline.base import PipelineStep, StepValidationError, SynthesisError
from yt_dbl.schemas import PipelineState, Segment, Speaker, StepName
from yt_dbl.utils.audio import get_audio_duration
from yt_dbl.utils.audio_processing import (
    SPEED_THRESHOLD,
    extract_voice_reference,
    postprocess_segment,
)
from yt_dbl.utils.languages import TTS_LANG_MAP
from yt_dbl.utils.logging import create_progress, log_info, suppress_library_noise

if TYPE_CHECKING:
    from pathlib import Path

SYNTH_META_FILE = "synth_meta.json"


# ── TTS generation ──────────────────────────────────────────────────────────


def _find_ref_text_for_speaker(
    segments: list[Segment],
    speaker: Speaker,
) -> str:
    """Find the original text of the speaker's reference segment."""
    for seg in segments:
        if (
            seg.speaker == speaker.id
            and seg.start <= speaker.reference_start
            and seg.end >= speaker.reference_end
        ):
            return seg.text
    # Fallback: first segment of that speaker
    for seg in segments:
        if seg.speaker == speaker.id:
            return seg.text
    return ""


class SynthesizeStep(PipelineStep):
    name = StepName.SYNTHESIZE
    description = "Synthesize speech (Qwen3-TTS voice cloning)"

    # ── validation ──────────────────────────────────────────────────────────

    def validate_inputs(self, state: PipelineState) -> None:
        if not state.segments:
            raise StepValidationError("No segments to synthesize")
        if not any(seg.translated_text for seg in state.segments):
            raise StepValidationError("Segments have no translated text")
        sep = state.get_step(StepName.SEPARATE)
        if "vocals" not in sep.outputs:
            raise StepValidationError("No vocals file from separation step")
        if not state.speakers:
            raise StepValidationError("No speakers detected — run transcribe first")

    # ── public API ──────────────────────────────────────────────────────────

    def run(self, state: PipelineState) -> PipelineState:
        meta_path = self.step_dir / SYNTH_META_FILE

        # Idempotency: reuse existing result
        if meta_path.exists():
            log_info("Found existing synthesis — loading from cache")
            return self._load_cached(state, meta_path)

        vocals_path = self._resolve_vocals(state)

        # Step 1: extract voice references for each speaker
        refs = self._extract_references(state, vocals_path)
        log_info(f"Extracted {len(refs)} voice references")

        # Step 2: synthesize segments via TTS
        self._synthesize_segments(state, refs)

        # Step 3: speed adjustment + normalisation
        self._postprocess_segments(state)

        # Step 4: clean up intermediate WAVs (raw_*, sped_*)
        self._cleanup_intermediates(state)

        # Persist metadata
        self._save_meta(state, meta_path)

        result = state.get_step(self.name)
        result.outputs = self._build_outputs(state)

        log_info(f"Synthesis complete: {len(state.segments)} segments")
        return state

    # ── internals ───────────────────────────────────────────────────────────

    def _resolve_vocals(self, state: PipelineState) -> Path:
        return self.resolve_step_file(state, StepName.SEPARATE, "vocals")

    def _extract_references(
        self,
        state: PipelineState,
        vocals_path: Path,
    ) -> dict[str, Path]:
        """Extract voice reference WAV for each speaker."""
        refs: dict[str, Path] = {}
        for speaker in state.speakers:
            ref_path = self.step_dir / f"ref_{speaker.id}.wav"
            if ref_path.exists():
                refs[speaker.id] = ref_path
                continue

            extract_voice_reference(
                vocals_path,
                speaker,
                ref_path,
                target_duration=self.settings.voice_ref_duration,
            )
            speaker.reference_path = ref_path.name
            refs[speaker.id] = ref_path
            log_info(
                f"  {speaker.id}: "
                f"{speaker.reference_start:.1f}-{speaker.reference_end:.1f}s "
                f"-> {ref_path.name}"
            )
        return refs

    def _synthesize_segments(
        self,
        state: PipelineState,
        refs: dict[str, Path],
    ) -> None:
        """Run TTS for each segment using speaker's voice reference."""
        model = self._load_tts_model()
        lang = TTS_LANG_MAP.get(state.target_language, "auto")
        total = len(state.segments)

        # Pre-compute ref text map to avoid O(N²) scans
        ref_text_map = {s.id: _find_ref_text_for_speaker(state.segments, s) for s in state.speakers}

        progress = create_progress()
        try:
            with progress:
                task = progress.add_task("  Synthesizing TTS", total=total)
                for seg in state.segments:
                    raw_path = self.step_dir / f"raw_{seg.id:04d}.wav"
                    if raw_path.exists():
                        seg.synth_path = raw_path.name
                        progress.advance(task)
                        continue

                    text = seg.translated_text or seg.text
                    ref_path = refs.get(seg.speaker)
                    ref_text = ref_text_map.get(seg.speaker, "")

                    audio = self._run_tts(model, text, ref_path, ref_text, lang)
                    self._save_wav(audio, raw_path, self.settings.tts_sample_rate)
                    seg.synth_path = raw_path.name
                    progress.advance(task)
        finally:
            # If not managed, free manually
            if self.model_manager is None:
                del model
                gc.collect()

    def _postprocess_segments(self, state: PipelineState) -> None:
        """Speed-adjust and normalize each synthesized segment.

        Segments are independent — each one gets its own ffmpeg subprocess
        chain (duration probe, optional speed-up, loudness normalisation).
        We run them in parallel via a thread pool; the GIL is released during
        subprocess I/O so all workers execute concurrently.
        """
        progress = create_progress()
        with progress:
            task = progress.add_task("  Postprocessing", total=len(state.segments))

            # Quick pass: skip already-done segments, collect work items
            to_process: list[Segment] = []
            for seg in state.segments:
                final_path = self.step_dir / f"segment_{seg.id:04d}.wav"
                raw_path = self.step_dir / f"raw_{seg.id:04d}.wav"

                if final_path.exists():
                    seg.synth_path = final_path.name
                    progress.advance(task)
                elif not raw_path.exists():
                    progress.advance(task)
                else:
                    to_process.append(seg)

            if not to_process:
                return

            # Parallel postprocessing: each segment is independent
            max_workers = min(os.cpu_count() or 1, len(to_process))
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(self._postprocess_one, seg): seg for seg in to_process}
                for future in as_completed(futures):
                    seg = futures[future]
                    synth_path, speed_factor = future.result()
                    seg.synth_path = synth_path
                    if speed_factor is not None:
                        seg.synth_speed_factor = speed_factor
                    progress.advance(task)

    def _postprocess_one(self, seg: Segment) -> tuple[str, float | None]:
        """Postprocess a single segment: speed-adjust + loudness-normalize + de-ess.

        Runs in a worker thread.  Uses ``postprocess_segment`` which
        combines filters into minimal ffmpeg calls, eliminating
        intermediate files.
        """
        raw_path = self.step_dir / f"raw_{seg.id:04d}.wav"
        final_path = self.step_dir / f"segment_{seg.id:04d}.wav"

        synth_dur = get_audio_duration(raw_path)
        original_dur = seg.duration
        speed: float | None = None

        if synth_dur > original_dur > 0:
            factor = min(synth_dur / original_dur, self.settings.max_speed_factor)
            if factor > SPEED_THRESHOLD:
                speed = factor

        postprocess_segment(raw_path, final_path, speed_factor=speed)

        if speed is not None:
            return final_path.name, round(speed, 3)
        return final_path.name, None

    def _cleanup_intermediates(self, state: PipelineState) -> None:
        """Remove raw_*.wav (and leftover sped_*/deessed_*) after postprocessing."""
        for seg in state.segments:
            final = self.step_dir / f"segment_{seg.id:04d}.wav"
            if not final.exists():
                continue
            for prefix in ("raw_", "sped_", "deessed_", "_sped_raw_"):
                tmp = self.step_dir / f"{prefix}{seg.id:04d}.wav"
                if tmp.exists():
                    tmp.unlink()

    # ── TTS model ───────────────────────────────────────────────────────────

    def _load_tts_model(self) -> Any:
        """Load Qwen3-TTS model via mlx-audio (through ModelManager if available)."""
        model_name = self.settings.tts_model

        if self.model_manager is not None:
            if model_name not in self.model_manager.registered_names:
                self.model_manager.register(
                    model_name,
                    loader=lambda name=model_name: self._load_tts_raw(name),
                )
            return self.model_manager.get(model_name)

        return self._load_tts_raw(model_name)

    @staticmethod
    def _load_tts_raw(model_name: str) -> Any:
        """Raw model loading without manager."""
        from mlx_audio.tts.utils import load_model

        log_info(f"Loading TTS model: {model_name}")
        with suppress_library_noise():
            return load_model(model_name)

    def _run_tts(
        self,
        model: Any,
        text: str,
        ref_audio: Path | None,
        ref_text: str,
        lang: str,
    ) -> Any:
        """Generate speech audio via Qwen3-TTS with voice cloning."""
        import mlx.core as mx

        kwargs: dict[str, Any] = {
            "text": text,
            "temperature": self.settings.tts_temperature,
            "top_k": self.settings.tts_top_k,
            "top_p": self.settings.tts_top_p,
            "repetition_penalty": self.settings.tts_repetition_penalty,
            "lang_code": lang,
        }
        if ref_audio is not None and ref_audio.exists():
            kwargs["ref_audio"] = str(ref_audio)
            if ref_text:
                kwargs["ref_text"] = ref_text

        results = list(model.generate(**kwargs))
        if not results:
            msg = f"TTS returned no audio for text: {text[:50]}"
            raise SynthesisError(msg)

        # Concatenate if multiple chunks
        audio_chunks = [r.audio for r in results]
        if len(audio_chunks) == 1:
            return audio_chunks[0]
        return mx.concatenate(audio_chunks)

    @staticmethod
    def _save_wav(audio: Any, path: Path, sample_rate: int) -> None:
        """Write an mlx array to a WAV file."""
        import numpy as np
        import soundfile as sf

        # Fast path: mlx array → numpy via the buffer protocol (zero-copy
        # when contiguous).  Falls back to np.asarray for plain arrays.
        arr = np.asarray(audio, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.squeeze()
        sf.write(str(path), arr, sample_rate)

    # ── Persistence ─────────────────────────────────────────────────────────

    @staticmethod
    def _build_outputs(state: PipelineState) -> dict[str, str]:
        outputs: dict[str, str] = {}
        for seg in state.segments:
            if seg.synth_path:
                outputs[f"seg_{seg.id}"] = seg.synth_path
        outputs["meta"] = SYNTH_META_FILE
        return outputs

    def _save_meta(self, state: PipelineState, path: Path) -> None:
        """Persist synthesis metadata for idempotent reloads."""
        data = {
            "segments": [
                {
                    "id": seg.id,
                    "synth_path": seg.synth_path,
                    "synth_speed_factor": seg.synth_speed_factor,
                }
                for seg in state.segments
            ],
            "speakers": [
                {
                    "id": s.id,
                    "reference_path": s.reference_path,
                }
                for s in state.speakers
            ],
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @staticmethod
    def _load_cached(state: PipelineState, meta_path: Path) -> PipelineState:
        """Load cached synthesis results from disk."""
        data = json.loads(meta_path.read_text(encoding="utf-8"))

        seg_lookup = {seg.id: seg for seg in state.segments}
        for item in data["segments"]:
            seg = seg_lookup.get(item["id"])
            if seg:
                seg.synth_path = item["synth_path"]
                seg.synth_speed_factor = item.get("synth_speed_factor", 1.0)

        spk_lookup = {s.id: s for s in state.speakers}
        for item in data["speakers"]:
            spk = spk_lookup.get(item["id"])
            if spk:
                spk.reference_path = item.get("reference_path", "")

        result = state.get_step(StepName.SYNTHESIZE)
        result.outputs = {}
        for seg in state.segments:
            if seg.synth_path:
                result.outputs[f"seg_{seg.id}"] = seg.synth_path
        result.outputs["meta"] = SYNTH_META_FILE

        return state
