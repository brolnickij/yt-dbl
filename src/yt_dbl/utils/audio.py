"""FFmpeg wrapper utilities."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def run_ffmpeg(args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run an ffmpeg command and return the result."""
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args]
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
        "-c:v",
        "copy",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
    ]
    if subtitle_path:
        args += ["-vf", f"subtitles={subtitle_path}"]
        # Need to re-encode video when burning subtitles
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
            "-vf",
            f"subtitles={subtitle_path}",
        ]
    else:
        args += ["-c:a", "aac"]
    args.append(str(output_path))
    run_ffmpeg(args)
    return output_path


def get_audio_duration(path: Path) -> float:
    """Get duration of an audio file in seconds."""
    result = subprocess.run(
        [
            "ffprobe",
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
