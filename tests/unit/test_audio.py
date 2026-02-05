"""Unit tests for yt_dbl.utils.audio — ffmpeg wrapper utilities."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from yt_dbl.utils.audio import extract_audio, get_audio_duration, replace_audio, run_ffmpeg

if TYPE_CHECKING:
    from pathlib import Path


class TestRunFfmpeg:
    @patch("subprocess.run")
    def test_basic_args(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
        run_ffmpeg(["-i", "in.mp4", "out.wav"])
        mock_run.assert_called_once_with(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", "in.mp4", "out.wav"],
            capture_output=True,
            text=True,
            check=True,
        )

    @patch("subprocess.run")
    def test_check_false(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=1)
        run_ffmpeg(["-i", "bad.mp4"], check=False)
        mock_run.assert_called_once()
        assert mock_run.call_args.kwargs["check"] is False

    @patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "ffmpeg"))
    def test_raises_on_error(self, mock_run: MagicMock) -> None:
        with pytest.raises(subprocess.CalledProcessError):
            run_ffmpeg(["-i", "bad.mp4", "out.wav"])


class TestExtractAudio:
    @patch("yt_dbl.utils.audio.run_ffmpeg")
    def test_default_params(self, mock_ff: MagicMock, tmp_path: Path) -> None:
        video = tmp_path / "video.mp4"
        audio = tmp_path / "audio.wav"
        result = extract_audio(video, audio)

        assert result == audio
        args = mock_ff.call_args[0][0]
        assert "-i" in args
        assert str(video) in args
        assert "-ac" in args  # mono
        assert "1" in args
        assert "-ar" in args
        assert "48000" in args
        assert "-vn" in args
        assert str(audio) in args

    @patch("yt_dbl.utils.audio.run_ffmpeg")
    def test_custom_sample_rate_stereo(self, mock_ff: MagicMock, tmp_path: Path) -> None:
        video = tmp_path / "video.mp4"
        audio = tmp_path / "audio.wav"
        extract_audio(video, audio, sample_rate=16000, mono=False)

        args = mock_ff.call_args[0][0]
        assert "16000" in args
        assert "-ac" not in args  # stereo — no -ac flag


class TestReplaceAudio:
    @patch("yt_dbl.utils.audio.run_ffmpeg")
    def test_without_subtitles(self, mock_ff: MagicMock, tmp_path: Path) -> None:
        video = tmp_path / "video.mp4"
        audio = tmp_path / "dubbed.wav"
        out = tmp_path / "final.mp4"
        result = replace_audio(video, audio, out)

        assert result == out
        args = mock_ff.call_args[0][0]
        assert "-c:v" in args
        assert "copy" in args
        assert "-c:a" in args
        assert "aac" in args
        # No -vf subtitles=... flag
        assert "-vf" not in args

    @patch("yt_dbl.utils.audio.run_ffmpeg")
    def test_with_subtitles(self, mock_ff: MagicMock, tmp_path: Path) -> None:
        video = tmp_path / "video.mp4"
        audio = tmp_path / "dubbed.wav"
        out = tmp_path / "final.mp4"
        subs = tmp_path / "subs.srt"
        replace_audio(video, audio, out, subtitle_path=subs)

        args = mock_ff.call_args[0][0]
        # When subtitles: no -c:v copy (need re-encode for burn-in)
        assert f"subtitles={subs}" in " ".join(args)


class TestGetAudioDuration:
    @patch("subprocess.run")
    def test_parses_duration(self, mock_run: MagicMock, tmp_path: Path) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="125.340000\n", stderr=""
        )
        duration = get_audio_duration(tmp_path / "audio.wav")
        assert duration == pytest.approx(125.34)

    @patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "ffprobe"))
    def test_raises_on_error(self, mock_run: MagicMock, tmp_path: Path) -> None:
        with pytest.raises(subprocess.CalledProcessError):
            get_audio_duration(tmp_path / "missing.wav")
