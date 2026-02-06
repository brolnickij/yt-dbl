"""Tests for yt_dbl.utils.audio_processing — ffmpeg-based audio post-processing."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING
from unittest.mock import patch

from yt_dbl.schemas import Speaker
from yt_dbl.utils.audio_processing import (
    deess,
    extract_voice_reference,
    normalize_loudness,
    speed_up_audio,
)

if TYPE_CHECKING:
    from pathlib import Path

# ── Voice reference tests ───────────────────────────────────────────────────


class TestExtractVoiceReference:
    def test_calls_ffmpeg(self, tmp_path: Path) -> None:
        speaker = Speaker(id="SPEAKER_00", reference_start=1.0, reference_end=5.0)
        output = tmp_path / "ref.wav"

        with patch("yt_dbl.utils.audio_processing.run_ffmpeg") as mock_ff:
            extract_voice_reference(tmp_path / "vocals.wav", speaker, output, target_duration=7.0)
            mock_ff.assert_called_once()
            args = mock_ff.call_args[0][0]
            assert "-ss" in args
            assert "1.0" in args
            assert "-t" in args
            assert "4.0" in args  # min(5.0-1.0, 7.0) = 4.0
            # Highpass + denoise filter chain
            assert "-af" in args
            af_idx = args.index("-af")
            assert "highpass=f=80" in args[af_idx + 1]
            assert "afftdn" in args[af_idx + 1]


# ── Speed adjustment tests ──────────────────────────────────────────────────


class TestSpeedUpAudio:
    def test_rubberband(self, tmp_path: Path) -> None:
        with (
            patch("yt_dbl.utils.audio_processing.has_rubberband", return_value=True),
            patch("yt_dbl.utils.audio_processing.run_ffmpeg") as mock_ff,
        ):
            speed_up_audio(tmp_path / "in.wav", tmp_path / "out.wav", 1.3)
            mock_ff.assert_called_once()
            args = mock_ff.call_args[0][0]
            assert any("rubberband" in a for a in args)
            assert any("pitch=1.0" in a for a in args)

    def test_atempo_fallback(self, tmp_path: Path) -> None:
        with (
            patch("yt_dbl.utils.audio_processing.has_rubberband", return_value=False),
            patch("yt_dbl.utils.audio_processing.run_ffmpeg") as mock_ff,
        ):
            speed_up_audio(tmp_path / "in.wav", tmp_path / "out.wav", 1.3)
            mock_ff.assert_called_once()
            args = mock_ff.call_args[0][0]
            assert any("atempo=1.3" in a for a in args)


# ── Normalisation tests ─────────────────────────────────────────────────────


class TestNormalizeLoudness:
    def test_two_pass(self, tmp_path: Path) -> None:
        """Two-pass loudnorm: pass 1 (measure, check=False) + pass 2 (apply)."""
        fake_measure = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="",
            stderr=(
                "[Parsed_loudnorm_0 @ 0x123] {\n"
                '    "input_i" : "-20.00",\n'
                '    "input_tp" : "-3.00",\n'
                '    "input_lra" : "5.00",\n'
                '    "input_thresh" : "-30.00",\n'
                '    "target_offset" : "4.00"\n'
                "}"
            ),
        )
        with patch(
            "yt_dbl.utils.audio_processing.run_ffmpeg", return_value=fake_measure
        ) as mock_ff:
            normalize_loudness(tmp_path / "in.wav", tmp_path / "out.wav")
            assert mock_ff.call_count == 2
            # Pass 1: measure
            pass1_args = mock_ff.call_args_list[0][0][0]
            assert any("loudnorm" in a for a in pass1_args)
            assert "/dev/null" in pass1_args
            # Pass 2: apply with measured values
            pass2_args = mock_ff.call_args_list[1][0][0]
            assert any("measured_I" in a for a in pass2_args)
            assert any("linear=true" in a for a in pass2_args)


# ── De-essing tests ─────────────────────────────────────────────────────────


class TestDeess:
    def test_calls_ffmpeg_with_highshelf_and_compand(self, tmp_path: Path) -> None:
        with patch("yt_dbl.utils.audio_processing.run_ffmpeg") as mock_ff:
            # Simulate successful output
            (tmp_path / "out.wav").write_bytes(b"fake")
            deess(tmp_path / "in.wav", tmp_path / "out.wav")
            mock_ff.assert_called_once()
            args = mock_ff.call_args[0][0]
            assert "-af" in args
            af_idx = args.index("-af")
            af = args[af_idx + 1]
            assert "highshelf" in af
            assert "compand" in af

    def test_fallback_copies_on_failure(self, tmp_path: Path) -> None:
        """If ffmpeg fails, input is copied to output."""
        src = tmp_path / "in.wav"
        src.write_bytes(b"audio data")
        dst = tmp_path / "out.wav"
        with patch("yt_dbl.utils.audio_processing.run_ffmpeg"):
            # Mock doesn't create output file -> fallback triggers
            deess(src, dst)
            assert dst.read_bytes() == b"audio data"
