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
    "deess",
    "extract_voice_reference",
    "normalize_loudness",
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
        _max_atempo = 100.0
        filters: list[str] = []
        remaining = factor
        while remaining > _max_atempo:
            filters.append(f"atempo={_max_atempo}")
            remaining /= _max_atempo
        filters.append(f"atempo={remaining:.4f}")

        run_ffmpeg(
            [
                "-i",
                str(input_path),
                "-filter:a",
                ",".join(filters),
                str(output_path),
            ]
        )
    return output_path


# ── De-essing ───────────────────────────────────────────────────────────────


def deess(input_path: Path, output_path: Path) -> Path:
    """Reduce harsh sibilants (s, sh, z) that TTS tends to over-emphasise.

    Applies a mild high-shelf compressor: frequencies above 4 kHz are
    dynamically attenuated when they exceed a threshold, taming sharp
    sibilance without dulling the entire mix.
    """
    run_ffmpeg(
        [
            "-i",
            str(input_path),
            "-af",
            "highshelf=gain=-3:frequency=4500:width_type=q:width=0.7,"
            "compand=attacks=0.005:decays=0.05:"
            "points=-80/-80|-6/-8|0/-3:volume=0",
            str(output_path),
        ],
        check=False,
    )
    # Fallback: if filter chain failed, copy input as-is
    if not output_path.exists():
        import shutil  # noqa: PLC0415

        shutil.copy2(input_path, output_path)
    return output_path


# ── Loudness normalisation ──────────────────────────────────────────────────


def normalize_loudness(input_path: Path, output_path: Path) -> Path:
    """Two-pass loudness normalisation to -16 LUFS.

    Pass 1 measures actual loudness; pass 2 applies correction with linear
    mode for minimal artefacts.  Upsamples to 48 kHz (pipeline standard).
    """
    # Pass 1: measure
    measure = run_ffmpeg(
        [
            "-i",
            str(input_path),
            "-filter:a",
            "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json",
            "-f",
            "null",
            "/dev/null",
        ],
        check=False,
    )

    # Try to parse measured values for precise second pass
    stderr = measure.stderr or ""
    # loudnorm prints JSON at the end of stderr
    json_match = re.search(r"\{[^}]+\}", stderr, re.DOTALL)
    if json_match:
        try:
            stats = json.loads(json_match.group())
            measured_i = stats["input_i"]
            measured_tp = stats["input_tp"]
            measured_lra = stats["input_lra"]
            measured_thresh = stats["input_thresh"]
            target_offset = stats["target_offset"]

            # Pass 2: apply with measured values (linear=true for best quality)
            run_ffmpeg(
                [
                    "-i",
                    str(input_path),
                    "-filter:a",
                    (
                        f"loudnorm=I=-16:TP=-1.5:LRA=11"
                        f":measured_I={measured_i}"
                        f":measured_TP={measured_tp}"
                        f":measured_LRA={measured_lra}"
                        f":measured_thresh={measured_thresh}"
                        f":offset={target_offset}"
                        f":linear=true"
                    ),
                    "-ar",
                    "48000",
                    str(output_path),
                ]
            )
        except (KeyError, json.JSONDecodeError):
            pass  # Fall through to single-pass
        else:
            return output_path

    # Fallback: single-pass loudnorm
    run_ffmpeg(
        [
            "-i",
            str(input_path),
            "-filter:a",
            "loudnorm=I=-16:TP=-1.5:LRA=11",
            "-ar",
            "48000",
            str(output_path),
        ]
    )
    return output_path
