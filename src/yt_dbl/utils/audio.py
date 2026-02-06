"""FFmpeg wrapper utilities."""

from __future__ import annotations

import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

__all__ = ["extract_audio", "get_audio_duration", "replace_audio", "run_ffmpeg"]

# Preferred ffmpeg-full path (brew keg-only install)
_FFMPEG_FULL_BIN = "/opt/homebrew/opt/ffmpeg-full/bin"


@lru_cache(maxsize=1)
def _detect_ffmpeg() -> str:
    """Auto-detect best ffmpeg binary, preferring ffmpeg-full for rubberband."""
    from yt_dbl.config import settings  # noqa: PLC0415

    if settings.ffmpeg_path:
        return settings.ffmpeg_path

    full = f"{_FFMPEG_FULL_BIN}/ffmpeg"
    if shutil.which(full):
        return full
    return "ffmpeg"


@lru_cache(maxsize=1)
def _detect_ffprobe() -> str:
    """Auto-detect ffprobe companion for the active ffmpeg."""
    ffmpeg = _detect_ffmpeg()
    if "/" in ffmpeg:
        probe = str(Path(ffmpeg).parent / "ffprobe")
        if shutil.which(probe):
            return probe
    return "ffprobe"


def has_rubberband() -> bool:
    """Check whether the active ffmpeg was compiled with librubberband."""
    try:
        result = subprocess.run(
            [_detect_ffmpeg(), "-filters"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return False
    else:
        return "rubberband" in result.stdout


def run_ffmpeg(args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run an ffmpeg command and return the result."""
    cmd = [_detect_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", *args]
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def extract_audio(
    video_path: Path,
    output_path: Path,
    sample_rate: int = 48000,
    mono: bool = True,
) -> Path:
    """Extract audio from video file as WAV."""
    args = ["-i", str(video_path)]
    if mono:
        args += ["-ac", "1"]
    args += ["-ar", str(sample_rate), "-vn", str(output_path)]
    run_ffmpeg(args)
    return output_path


def replace_audio(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    subtitle_path: Path | None = None,
) -> Path:
    """Replace audio track in video, optionally burn in subtitles."""
    args = [
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:a",
        "aac",
        "-b:a",
        "320k",  # High-quality AAC
    ]
    if subtitle_path:
        # Need to re-encode video when burning subtitles
        args += ["-vf", f"subtitles={subtitle_path}"]
    else:
        # No subtitles — copy video stream as-is
        args += ["-c:v", "copy"]
    args.append(str(output_path))
    run_ffmpeg(args)
    return output_path


def get_audio_duration(path: Path) -> float:
    """Get duration of an audio file in seconds."""
    result = subprocess.run(
        [
            _detect_ffprobe(),
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())
