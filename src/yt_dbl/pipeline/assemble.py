"""Step 6: Assemble final video with dubbed audio.

Builds a continuous speech track from synthesized WAV segments (placed at
original timecodes with short crossfades), mixes it with the background
music track, and muxes the result onto the original video with optional
soft-subtitles — all in a single ffmpeg call.
"""

from __future__ import annotations

from math import gcd
from typing import TYPE_CHECKING

from yt_dbl.pipeline.base import AssemblyError, PipelineStep, StepValidationError
from yt_dbl.schemas import STEP_DIRS, PipelineState, Segment, StepName, StepResult
from yt_dbl.utils.audio import get_audio_duration, run_ffmpeg
from yt_dbl.utils.logging import log_info, log_warning

if TYPE_CHECKING:
    from pathlib import Path

SPEECH_TRACK_FILE = "speech.pcm"
CROSSFADE_MS = 50
_SPEECH_CHUNK_SECONDS = 30


# ── Speech track assembly ───────────────────────────────────────────────────


def _collect_segment_entries(
    segments: list[Segment],
    synth_dir: Path,
    sample_rate: int,
) -> list[tuple[int, int, Path]]:
    """Collect ``(start_sample, end_sample, path)`` from valid segments.

    Only reads WAV headers (``sf.info``) — no audio data is loaded.
    Logs a warning listing IDs of any segments that are skipped (no
    synthesis output or missing file).
    """
    import soundfile as sf

    entries: list[tuple[int, int, Path]] = []
    skipped: list[int] = []
    for seg in segments:
        if not seg.synth_path:
            skipped.append(seg.id)
            continue
        wav_path = synth_dir / seg.synth_path
        if not wav_path.exists():
            skipped.append(seg.id)
            continue
        info = sf.info(str(wav_path))
        n_frames = info.frames
        if info.samplerate != sample_rate:
            n_frames = int(n_frames * sample_rate / info.samplerate)
        start_sample = int(seg.start * sample_rate)
        entries.append((start_sample, start_sample + n_frames, wav_path))

    if skipped:
        log_warning(f"{len(skipped)} segment(s) missing from speech track: {skipped}")

    entries.sort()
    return entries


def _build_speech_track(
    segments: list[Segment],
    synth_dir: Path,
    output_path: Path,
    total_duration: float,
    sample_rate: int = 48000,
    crossfade_ms: int = CROSSFADE_MS,
) -> int:
    """Build a continuous speech track by streaming chunks to raw PCM.

    Instead of allocating a single numpy array for the full timeline
    (which grows to ~2 GB for a 3-hour video at 48 kHz), the audio is
    processed in fixed-size windows (~30 s) and each window is flushed
    directly to disk.

    The output is **raw 32-bit float little-endian PCM** (``f32le``) —
    no container header and no file-size limit.

    Returns the total number of samples written.
    """
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    total_samples = int(total_duration * sample_rate) + sample_rate  # +1 s safety
    chunk_samples = _SPEECH_CHUNK_SECONDS * sample_rate
    fade_samples = int(crossfade_ms / 1000 * sample_rate)

    entries = _collect_segment_entries(segments, synth_dir, sample_rate)

    # Pre-load and prepare all segment audio once (read + resample + fade).
    # This avoids re-reading WAVs for segments that span chunk boundaries.
    seg_audio: dict[Path, np.ndarray] = {}
    for _seg_start, _seg_end, wav_path in entries:
        if wav_path in seg_audio:
            continue
        data, sr = sf.read(str(wav_path), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)  # stereo → mono
        if sr != sample_rate:
            up = sample_rate // gcd(sample_rate, sr)
            down = sr // gcd(sample_rate, sr)
            data = resample_poly(data, up, down).astype(np.float32)
        if fade_samples > 0 and len(data) > fade_samples * 2:
            t = np.linspace(0.0, np.pi / 2, fade_samples, dtype=np.float32)
            data[:fade_samples] *= np.sin(t)
            data[-fade_samples:] *= np.cos(t)
        seg_audio[wav_path] = data

    seg_ptr = 0
    samples_written = 0

    with output_path.open("wb") as out:
        for chunk_offset in range(0, total_samples, chunk_samples):
            chunk_len = min(chunk_samples, total_samples - chunk_offset)
            chunk = np.zeros(chunk_len, dtype=np.float32)
            chunk_end = chunk_offset + chunk_len

            # Advance past segments that end before this chunk
            while seg_ptr < len(entries) and entries[seg_ptr][1] <= chunk_offset:
                seg_ptr += 1

            for i in range(seg_ptr, len(entries)):
                seg_start, seg_end, wav_path = entries[i]
                if seg_start >= chunk_end:
                    break
                if seg_end <= chunk_offset:
                    continue  # short segment between seg_ptr and a longer one

                data = seg_audio[wav_path]

                # Clip to total_samples boundary
                if seg_start + len(data) > total_samples:
                    data = data[: total_samples - seg_start]

                # Mix overlapping region into chunk
                src_start = max(0, chunk_offset - seg_start)
                src_end = min(len(data), chunk_end - seg_start)
                dst_start = max(0, seg_start - chunk_offset)
                n = src_end - src_start
                if n > 0:
                    chunk[dst_start : dst_start + n] += data[src_start:src_end]

            out.write(chunk.tobytes())
            samples_written += chunk_len

    return samples_written


# ── FFmpeg final assembly ───────────────────────────────────────────────────


def _assemble_video(
    video_path: Path,
    speech_path: Path,
    background_path: Path,
    output_path: Path,
    background_volume: float,
    sample_rate: int = 48000,
    subtitle_path: Path | None = None,
    subtitle_mode: str = "softsub",
    output_format: str = "mp4",
    background_ducking: bool = True,
) -> Path:
    """Combine video + speech + background [+ subtitles] in one ffmpeg call.

    Audio pipeline:
      background → volume → (optional sidechain duck) → mix with speech
      → limiter → AAC 320 kbps

    When *background_ducking* is enabled the background is compressed
    (ducked) whenever speech is present, reducing interference from
    imperfect vocal separation.

    Video stream is copied without re-encoding (unless hardsub mode).
    """
    inputs = [
        "-i",
        str(video_path),
        # Speech track is raw f32le PCM — tell ffmpeg the format
        "-f",
        "f32le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-i",
        str(speech_path),
        "-i",
        str(background_path),
    ]

    has_subs = subtitle_path is not None and subtitle_path.exists() and subtitle_mode != "none"
    if has_subs:
        inputs += ["-i", str(subtitle_path)]

    # Audio filter: mix speech + attenuated background
    if background_ducking:
        # Sidechain ducking: reduce background when speech is active.
        # asplit clones the speech stream — one copy feeds the sidechain
        # detector, the other is mixed with the ducked background.
        filter_parts = [
            "[1:a]asplit[speech][sc]",
            f"[2:a]volume={background_volume:.2f}[bg]",
            ("[bg][sc]sidechaincompress=threshold=0.01:ratio=6:attack=50:release=800[ducked]"),
            (
                "[speech][ducked]amix=inputs=2:duration=longest:normalize=0,"
                "alimiter=limit=0.95:level=false[mixed]"
            ),
        ]
    else:
        filter_parts = [
            f"[2:a]volume={background_volume:.2f}[bg]",
            (
                "[1:a][bg]amix=inputs=2:duration=longest:normalize=0,"
                "alimiter=limit=0.95:level=false[mixed]"
            ),
        ]

    # Hardsub: burn subtitles into video (requires re-encode)
    if has_subs and subtitle_mode == "hardsub":
        escaped = str(subtitle_path).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
        filter_parts.append(f"[0:v]subtitles='{escaped}'[vout]")
        video_map = "[vout]"
        video_codec = ["-c:v", "libx264", "-crf", "18", "-preset", "medium"]
    else:
        video_map = "0:v:0"
        video_codec = ["-c:v", "copy"]

    filter_complex = ";".join(filter_parts)

    args = [
        *inputs,
        "-filter_complex",
        filter_complex,
        "-map",
        video_map,
        "-map",
        "[mixed]",
    ]

    # Softsub: add subtitle stream
    if has_subs and subtitle_mode == "softsub":
        args += ["-map", "3:s:0"]
        if output_format == "mp4":
            args += ["-c:s", "mov_text"]
        else:
            args += ["-c:s", "srt"]

    args += [
        *video_codec,
        "-c:a",
        "aac",
        "-b:a",
        "320k",
        "-ar",
        "48000",
        str(output_path),
    ]

    try:
        run_ffmpeg(args)
    except Exception as exc:
        raise AssemblyError(f"ffmpeg assembly failed: {exc}") from exc
    return output_path


# ── Step ────────────────────────────────────────────────────────────────────


class AssembleStep(PipelineStep):
    name = StepName.ASSEMBLE
    description = "Assemble final video with dubbed audio"

    # ── validation ──────────────────────────────────────────────────────────

    def validate_inputs(self, state: PipelineState) -> None:
        synth = state.get_step(StepName.SYNTHESIZE)
        if not synth.outputs:
            raise StepValidationError("No synthesized segments")
        dl = state.get_step(StepName.DOWNLOAD)
        if "video" not in dl.outputs:
            raise StepValidationError("No video file from download step")
        sep = state.get_step(StepName.SEPARATE)
        if "background" not in sep.outputs:
            raise StepValidationError("No background audio from separation step")

    # ── public API ──────────────────────────────────────────────────────────

    def run(self, state: PipelineState) -> PipelineState:
        output_name = f"result.{self.settings.output_format}"
        # Write final result to job dir (parent of step dirs) for easy access
        job_dir = self.step_dir.parent
        output_path = job_dir / output_name

        # Idempotency: skip if result already exists
        if output_path.exists():
            log_info("Found existing result — skipping assembly")
            result = state.get_step(self.name)
            result.outputs = {"result": output_name}
            self._add_subtitle_output(state, result)
            return state

        # Resolve input paths
        video_path = self._resolve_step_output(state, StepName.DOWNLOAD, "video")
        background_path = self._resolve_step_output(state, StepName.SEPARATE, "background")
        synth_dir = self.settings.step_dir(state.video_id, STEP_DIRS[StepName.SYNTHESIZE])
        subtitle_path = self._resolve_subtitle_path(state)

        # Step 1: Build unified speech track
        log_info("Building speech track from segments...")
        total_duration = self._get_total_duration(state, video_path)
        speech_path = self.step_dir / SPEECH_TRACK_FILE
        n_samples = _build_speech_track(
            state.segments,
            synth_dir,
            speech_path,
            total_duration,
            sample_rate=self.settings.sample_rate,
        )
        speech_dur = n_samples / self.settings.sample_rate
        log_info(f"Speech track: {speech_dur:.1f}s ({len(state.segments)} segments)")

        # Step 2: Assemble final video via ffmpeg
        log_info("Assembling final video...")
        _assemble_video(
            video_path=video_path,
            speech_path=speech_path,
            background_path=background_path,
            output_path=output_path,
            background_volume=self.settings.background_volume,
            sample_rate=self.settings.sample_rate,
            subtitle_path=subtitle_path,
            subtitle_mode=self.settings.subtitle_mode,
            output_format=self.settings.output_format,
            background_ducking=self.settings.background_ducking,
        )

        result = state.get_step(self.name)
        result.outputs = {"result": output_name}
        self._add_subtitle_output(state, result)

        log_info(f"Assembly complete → {output_path}")
        return state

    # ── internals ───────────────────────────────────────────────────────────

    def _resolve_step_output(self, state: PipelineState, step: StepName, key: str) -> Path:
        """Resolve an output file path from a previous pipeline step."""
        outputs = state.get_step(step).outputs
        step_dir = self.settings.step_dir(state.video_id, STEP_DIRS[step])
        return step_dir / outputs[key]

    def _resolve_subtitle_path(self, state: PipelineState) -> Path | None:
        """Find the SRT file produced by the translate step."""
        trans = state.get_step(StepName.TRANSLATE)
        if "subtitles" not in trans.outputs:
            return None
        trans_dir = self.settings.step_dir(state.video_id, STEP_DIRS[StepName.TRANSLATE])
        srt_path = trans_dir / trans.outputs["subtitles"]
        return srt_path if srt_path.exists() else None

    def _get_total_duration(self, state: PipelineState, video_path: Path) -> float:
        """Determine the target duration for the speech track."""
        # Prefer video metadata
        if state.meta and state.meta.duration > 0:
            return state.meta.duration
        # Fallback: probe the video file
        try:
            return get_audio_duration(video_path)
        except Exception:
            log_warning(f"Could not probe duration of {video_path.name}, using fallback")
        # Last resort: use the end of the last segment + buffer
        if state.segments:
            return max(seg.end for seg in state.segments) + 1.0
        return 60.0

    @staticmethod
    def _add_subtitle_output(
        state: PipelineState,
        result: StepResult,
    ) -> None:
        """Copy subtitle reference from translate step into outputs."""
        trans = state.get_step(StepName.TRANSLATE)
        if "subtitles" in trans.outputs:
            result.outputs["subtitles"] = trans.outputs["subtitles"]
