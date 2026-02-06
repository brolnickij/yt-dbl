"""Unit tests for yt_dbl.utils.audio — ffmpeg wrapper utilities."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from yt_dbl.utils.audio import (
    _detect_ffmpeg,
    _detect_ffprobe,
    extract_audio,
    get_audio_duration,
    has_rubberband,
    run_ffmpeg,
    set_ffmpeg_path,
)

if TYPE_CHECKING:
    from pathlib import Path


# Reset lru_cache and override between tests to avoid stale state
@pytest.fixture(autouse=True)
def _clear_detection_cache() -> None:
    set_ffmpeg_path("")
    _detect_ffprobe.cache_clear()
    has_rubberband.cache_clear()


class TestFfmpegDetection:
    @patch("shutil.which", return_value="/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg")
    def test_prefers_ffmpeg_full(self, mock_which: MagicMock) -> None:
        result = _detect_ffmpeg()
        assert "ffmpeg-full" in result

    def test_uses_explicit_path(self) -> None:
        set_ffmpeg_path("/custom/ffmpeg")
        result = _detect_ffmpeg()
        assert result == "/custom/ffmpeg"

    @patch("shutil.which", return_value=None)
    def test_falls_back_to_ffmpeg(self, mock_which: MagicMock) -> None:
        result = _detect_ffmpeg()
        assert result == "ffmpeg"

    @patch(
        "yt_dbl.utils.audio._detect_ffmpeg",
        return_value="/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg",
    )
    @patch("shutil.which", return_value="/opt/homebrew/opt/ffmpeg-full/bin/ffprobe")
    def test_ffprobe_companion(self, mock_which: MagicMock, mock_detect: MagicMock) -> None:
        _detect_ffprobe.cache_clear()
        result = _detect_ffprobe()
        assert "ffprobe" in result
        assert "ffmpeg-full" in result


class TestHasRubberband:
    @patch("yt_dbl.utils.audio._detect_ffmpeg", return_value="ffmpeg")
    @patch("subprocess.run")
    def test_returns_true(self, mock_run: MagicMock, mock_detect: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="rubberband  A->A  Apply time-stretching"
        )
        assert has_rubberband() is True

    @patch("yt_dbl.utils.audio._detect_ffmpeg", return_value="ffmpeg")
    @patch("subprocess.run")
    def test_returns_false(self, mock_run: MagicMock, mock_detect: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="atempo  A->A  Adjust audio tempo"
        )
        assert has_rubberband() is False

    @patch("yt_dbl.utils.audio._detect_ffmpeg", return_value="ffmpeg")
    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_returns_false_on_missing(self, mock_run: MagicMock, mock_detect: MagicMock) -> None:
        assert has_rubberband() is False


class TestRunFfmpeg:
    @patch("yt_dbl.utils.audio._detect_ffmpeg", return_value="ffmpeg")
    @patch("subprocess.run")
    def test_basic_args(self, mock_run: MagicMock, mock_detect: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
        run_ffmpeg(["-i", "in.mp4", "out.wav"])

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        # Binary is first; user args are appended after default flags
        assert cmd[0] == "ffmpeg"
        assert cmd[-3:] == ["-i", "in.mp4", "out.wav"]
        for flag in ("-y", "-hide_banner"):
            assert flag in cmd
        assert mock_run.call_args[1] == {"capture_output": True, "text": True, "check": True}

    @patch("yt_dbl.utils.audio._detect_ffmpeg", return_value="ffmpeg")
    @patch("subprocess.run")
    def test_check_false(self, mock_run: MagicMock, mock_detect: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=1)
        run_ffmpeg(["-i", "bad.mp4"], check=False)
        mock_run.assert_called_once()
        assert mock_run.call_args.kwargs["check"] is False

    @patch("yt_dbl.utils.audio._detect_ffmpeg", return_value="ffmpeg")
    @patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "ffmpeg"))
    def test_raises_on_error(self, mock_run: MagicMock, mock_detect: MagicMock) -> None:
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


class TestGetAudioDuration:
    @patch("yt_dbl.utils.audio._detect_ffprobe", return_value="ffprobe")
    @patch("subprocess.run")
    def test_parses_duration(
        self, mock_run: MagicMock, mock_probe: MagicMock, tmp_path: Path
    ) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="125.340000\n", stderr=""
        )
        duration = get_audio_duration(tmp_path / "audio.wav")
        assert duration == pytest.approx(125.34)

    @patch("yt_dbl.utils.audio._detect_ffprobe", return_value="ffprobe")
    @patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "ffprobe"))
    def test_raises_on_error(
        self, mock_run: MagicMock, mock_probe: MagicMock, tmp_path: Path
    ) -> None:
        with pytest.raises(subprocess.CalledProcessError):
            get_audio_duration(tmp_path / "missing.wav")
