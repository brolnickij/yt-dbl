"""FFmpeg wrapper utilities."""

from __future__ import annotations

import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

__all__ = ["extract_audio", "get_audio_duration", "run_ffmpeg", "set_ffmpeg_path"]

# Preferred ffmpeg-full path (brew keg-only install)
_FFMPEG_FULL_BIN = "/opt/homebrew/opt/ffmpeg-full/bin"

# Module-level override set via set_ffmpeg_path()
_ffmpeg_override: str = ""


def set_ffmpeg_path(path: str) -> None:
    """Set explicit ffmpeg binary path and clear detection caches."""
    global _ffmpeg_override  # noqa: PLW0603
    _ffmpeg_override = path
    _detect_ffmpeg.cache_clear()
    _detect_ffprobe.cache_clear()
    has_rubberband.cache_clear()


@lru_cache(maxsize=1)
def _detect_ffmpeg() -> str:
    """Auto-detect best ffmpeg binary, preferring ffmpeg-full for rubberband."""
    if _ffmpeg_override:
        return _ffmpeg_override

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


@lru_cache(maxsize=1)
def has_rubberband() -> bool:
    """Check whether the active ffmpeg was compiled with librubberband."""
    try:
        result = subprocess.run(
            [_detect_ffmpeg(), "-filters"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    else:
        return "rubberband" in result.stdout


def run_ffmpeg(
    args: list[str],
    check: bool = True,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run an ffmpeg command and return the result.

    Parameters
    ----------
    args
        FFmpeg arguments (without the binary name).
    check
        Raise ``CalledProcessError`` on non-zero exit.
    timeout
        Maximum seconds to wait.  ``None`` means no limit.  When
        exceeded, ``subprocess.TimeoutExpired`` is raised.
    """
    cmd = [_detect_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", *args]
    return subprocess.run(cmd, capture_output=True, text=True, check=check, timeout=timeout)


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
        timeout=30,
    )
    return float(result.stdout.strip())
