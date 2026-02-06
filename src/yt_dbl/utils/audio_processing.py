"""Audio post-processing utilities for synthesized speech.

Reusable ffmpeg-based helpers: voice-reference extraction, speed adjustment,
de-essing, and loudness normalisation.  All functions are pure (no class
state) and depend only on ``utils.audio`` for the ffmpeg wrapper.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from yt_dbl.utils.audio import has_rubberband, run_ffmpeg

if TYPE_CHECKING:
    from pathlib import Path

    from yt_dbl.schemas import Speaker

__all__ = [
    "extract_voice_reference",
    "postprocess_segment",
    "speed_up_audio",
]

# ── Voice reference extraction ──────────────────────────────────────────────


def extract_voice_reference(
    vocals_path: Path,
    speaker: Speaker,
    output_path: Path,
    target_duration: float,
) -> Path:
    """Extract a voice reference clip for a speaker from vocals.wav.

    Uses ffmpeg to cut the best segment (highest-scoring continuous speech
    by this speaker), applies a highpass filter at 80 Hz to remove rumble
    and ``afftdn`` to reduce residual noise — both improve TTS cloning.
    Clips to *target_duration* seconds max.
    """
    start = speaker.reference_start
    duration = min(speaker.reference_end - start, target_duration)

    run_ffmpeg(
        [
            "-i",
            str(vocals_path),
            "-ss",
            str(start),
            "-t",
            str(duration),
            "-af",
            "highpass=f=80,afftdn=nf=-25",
            "-ac",
            "1",
            "-ar",
            "24000",  # Qwen3-TTS internal sample rate for best cloning
            str(output_path),
        ]
    )
    return output_path


# ── Speed adjustment via ffmpeg ─────────────────────────────────────────────


def speed_up_audio(
    input_path: Path,
    output_path: Path,
    factor: float,
) -> Path:
    """Speed up audio by *factor*, preserving pitch and timbre.

    Uses librubberband (pitch-preserving time-stretch) when available,
    falls back to ffmpeg atempo filter otherwise.
    """
    if has_rubberband():
        # rubberband preserves pitch: tempo=2.0 means 2x faster
        run_ffmpeg(
            [
                "-i",
                str(input_path),
                "-filter:a",
                f"rubberband=tempo={factor:.4f}:pitch=1.0",
                str(output_path),
            ]
        )
    else:
        # Fallback: atempo (0.5-100.0 range per filter)
        run_ffmpeg(
            [
                "-i",
                str(input_path),
                "-filter:a",
                _atempo_chain(factor),
                str(output_path),
            ]
        )
    return output_path


# ── Combined postprocessing ──────────────────────────────────────────────────

_DEESS_FILTER = (
    "highshelf=gain=-3:frequency=4500:width_type=q:width=0.7,"
    "compand=attacks=0.005:decays=0.05:"
    "points=-80/-80|-6/-8|0/-3:volume=0"
)


def _atempo_chain(factor: float) -> str:
    """Build an atempo filter chain for the given speed factor."""
    _max_atempo = 100.0
    filters: list[str] = []
    remaining = factor
    while remaining > _max_atempo:
        filters.append(f"atempo={_max_atempo}")
        remaining /= _max_atempo
    filters.append(f"atempo={remaining:.4f}")
    return ",".join(filters)


def _parse_loudnorm_stats(stderr: str) -> dict[str, str] | None:
    """Parse loudnorm measurement JSON from ffmpeg stderr."""
    json_match = re.search(r"\{[^}]+\}", stderr, re.DOTALL)
    if not json_match:
        return None
    try:
        stats: dict[str, str] = json.loads(json_match.group())
        required = ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")
        if all(k in stats for k in required):
            return stats
    except json.JSONDecodeError:
        pass
    return None


def _loudnorm_apply_filter(stats: dict[str, str] | None) -> str:
    """Build the loudnorm apply filter (pass 2) from measured stats."""
    if stats:
        return (
            f"loudnorm=I=-16:TP=-1.5:LRA=11"
            f":measured_I={stats['input_i']}"
            f":measured_TP={stats['input_tp']}"
            f":measured_LRA={stats['input_lra']}"
            f":measured_thresh={stats['input_thresh']}"
            f":offset={stats['target_offset']}"
            f":linear=true"
        )
    return "loudnorm=I=-16:TP=-1.5:LRA=11"


def postprocess_segment(
    input_path: Path,
    output_path: Path,
    speed_factor: float | None = None,
) -> Path:
    """Postprocess a synthesized WAV: optional speed-up + loudnorm + de-ess.

    Combines filters to minimise ffmpeg calls and intermediate files:

    - **No speed:**  2 calls (measure → apply+deess), 0 temp files.
    - **rubberband:** 3 calls (speed → measure → apply+deess), 1 temp file
      (cleaned automatically).
    - **atempo:**    2 calls (speed+measure → speed+apply+deess), 0 temp files
      (atempo is cheap enough to run twice).
    """
    _speed_threshold = 1.01
    needs_speed = speed_factor is not None and speed_factor > _speed_threshold

    speed_prefix = ""
    measure_input = input_path
    apply_input = input_path
    temp_sped: Path | None = None

    if needs_speed:
        if speed_factor is None:  # pragma: no cover — guarded by needs_speed
            msg = "speed_factor must not be None when needs_speed is True"
            raise ValueError(msg)
        if has_rubberband():
            # rubberband is expensive — materialise once
            temp_sped = input_path.parent / f"_sped_{input_path.stem}.wav"
            speed_up_audio(input_path, temp_sped, speed_factor)
            measure_input = temp_sped
            apply_input = temp_sped
        else:
            # atempo is cheap — include in both passes
            speed_prefix = _atempo_chain(speed_factor) + ","

    # Pass 1: measure loudness
    measure_filter = f"{speed_prefix}loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json"
    measure = run_ffmpeg(
        ["-i", str(measure_input), "-filter:a", measure_filter, "-f", "null", "/dev/null"],
        check=False,
    )

    stats = _parse_loudnorm_stats(measure.stderr or "")
    loudnorm = _loudnorm_apply_filter(stats)

    # Pass 2: apply [speed +] loudnorm + deess in one call
    apply_filter = f"{speed_prefix}{loudnorm},{_DEESS_FILTER}"
    run_ffmpeg(
        ["-i", str(apply_input), "-filter:a", apply_filter, "-ar", "48000", str(output_path)],
        check=False,
    )

    # Fallback: if combined filter failed, try without deess
    if not output_path.exists():
        fb_filter = f"{speed_prefix}{loudnorm}"
        run_ffmpeg(
            ["-i", str(apply_input), "-filter:a", fb_filter, "-ar", "48000", str(output_path)],
        )

    # Clean up rubberband temp
    if temp_sped is not None and temp_sped.exists():
        temp_sped.unlink()

    return output_path
