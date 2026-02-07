"""Tests for yt_dbl.utils.audio_processing — ffmpeg-based audio post-processing."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING
from unittest.mock import patch

from yt_dbl.schemas import Speaker
from yt_dbl.utils.audio_processing import (
    extract_voice_reference,
    postprocess_segment,
    speed_up_audio,
)

if TYPE_CHECKING:
    from pathlib import Path


# ── Helpers ─────────────────────────────────────────────────────────────────


def _ok_result() -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")


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


# ── postprocess_segment tests ───────────────────────────────────────────────


def _ffmpeg_touch_side_effect(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Side effect that creates the output file (last positional arg)."""
    cmd_args = args[0] if args else kwargs.get("args", [])
    if cmd_args:
        candidate = cmd_args[-1]
        if not candidate.startswith("-"):
            from pathlib import Path as _Path

            _Path(candidate).parent.mkdir(parents=True, exist_ok=True)
            _Path(candidate).write_bytes(b"fake-audio")
    return _ok_result()


class TestPostprocessSegment:
    def test_no_speed_one_call(self, tmp_path: Path) -> None:
        """Without speed: 1 ffmpeg call (single-pass loudnorm + deess)."""
        src = tmp_path / "raw.wav"
        dst = tmp_path / "out.wav"
        src.write_bytes(b"fake")

        with patch(
            "yt_dbl.utils.audio_processing.run_ffmpeg",
            side_effect=_ffmpeg_touch_side_effect,
        ) as mock_ff:
            postprocess_segment(src, dst)
            assert mock_ff.call_count == 1
            call_args = mock_ff.call_args_list[0][0][0]
            assert any("loudnorm" in a for a in call_args)
            assert any("highshelf" in a for a in call_args)
            assert any("compand" in a for a in call_args)

    def test_atempo_speed_one_call(self, tmp_path: Path) -> None:
        """With atempo speed: 1 ffmpeg call (atempo + loudnorm + deess)."""
        src = tmp_path / "raw.wav"
        dst = tmp_path / "out.wav"
        src.write_bytes(b"fake")

        with (
            patch("yt_dbl.utils.audio_processing.has_rubberband", return_value=False),
            patch(
                "yt_dbl.utils.audio_processing.run_ffmpeg",
                side_effect=_ffmpeg_touch_side_effect,
            ) as mock_ff,
        ):
            postprocess_segment(src, dst, speed_factor=1.3)
            assert mock_ff.call_count == 1
            call_args = mock_ff.call_args_list[0][0][0]
            filter_str = " ".join(call_args)
            assert "atempo" in filter_str
            assert "loudnorm" in filter_str

    def test_rubberband_speed_two_calls(self, tmp_path: Path) -> None:
        """With rubberband speed: 2 ffmpeg calls (speed + loudnorm+deess)."""
        src = tmp_path / "raw.wav"
        dst = tmp_path / "out.wav"
        src.write_bytes(b"fake")

        with (
            patch("yt_dbl.utils.audio_processing.has_rubberband", return_value=True),
            patch(
                "yt_dbl.utils.audio_processing.run_ffmpeg",
                side_effect=_ffmpeg_touch_side_effect,
            ) as mock_ff,
        ):
            postprocess_segment(src, dst, speed_factor=1.3)
            assert mock_ff.call_count == 2
            # First call: rubberband
            rb_args = mock_ff.call_args_list[0][0][0]
            assert any("rubberband" in a for a in rb_args)

    def test_no_speed_factor_ignored(self, tmp_path: Path) -> None:
        """speed_factor=None or <=1.01 should not trigger speed filters."""
        src = tmp_path / "raw.wav"
        dst = tmp_path / "out.wav"
        src.write_bytes(b"fake")

        with patch(
            "yt_dbl.utils.audio_processing.run_ffmpeg",
            side_effect=_ffmpeg_touch_side_effect,
        ) as mock_ff:
            postprocess_segment(src, dst, speed_factor=1.005)
            assert mock_ff.call_count == 1
            filter_arg = " ".join(mock_ff.call_args_list[0][0][0])
            assert "atempo" not in filter_arg
            assert "rubberband" not in filter_arg

    def test_rubberband_temp_cleaned(self, tmp_path: Path) -> None:
        """Rubberband temporary file is deleted after processing."""
        src = tmp_path / "raw.wav"
        dst = tmp_path / "out.wav"
        src.write_bytes(b"fake")

        with (
            patch("yt_dbl.utils.audio_processing.has_rubberband", return_value=True),
            patch(
                "yt_dbl.utils.audio_processing.run_ffmpeg",
                side_effect=_ffmpeg_touch_side_effect,
            ),
        ):
            postprocess_segment(src, dst, speed_factor=1.5)
            # Temp file should be removed
            temp_sped = tmp_path / "_sped_raw.wav"
            assert not temp_sped.exists()

    def test_empty_output_triggers_fallback(self, tmp_path: Path) -> None:
        """When the primary call creates a 0-byte file, fallback re-runs without deess."""
        src = tmp_path / "raw.wav"
        dst = tmp_path / "out.wav"
        src.write_bytes(b"fake")

        call_count = 0

        def _empty_then_ok(*args, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal call_count
            call_count += 1
            cmd_args = args[0] if args else kwargs.get("args", [])
            if cmd_args:
                candidate = cmd_args[-1]
                if not candidate.startswith("-"):
                    from pathlib import Path as _Path

                    _Path(candidate).parent.mkdir(parents=True, exist_ok=True)
                    if call_count == 1:
                        # Primary call: create EMPTY file → should trigger fallback
                        _Path(candidate).write_bytes(b"")
                    else:
                        # Fallback: create proper output
                        _Path(candidate).write_bytes(b"fallback-ok")
            return _ok_result()

        with patch(
            "yt_dbl.utils.audio_processing.run_ffmpeg",
            side_effect=_empty_then_ok,
        ) as mock_ff:
            postprocess_segment(src, dst)
            assert mock_ff.call_count == 2  # primary + fallback
            assert dst.exists()
            assert dst.stat().st_size > 0
            # Fallback should not contain deess filters
            fallback_args = mock_ff.call_args_list[1][0][0]
            filter_str = " ".join(fallback_args)
            assert "loudnorm" in filter_str
            assert "highshelf" not in filter_str
            assert "compand" not in filter_str

    def test_deess_failure_falls_back_to_loudnorm_only(self, tmp_path: Path) -> None:
        """When combined loudnorm+deess fails, falls back to loudnorm only."""
        src = tmp_path / "raw.wav"
        dst = tmp_path / "out.wav"
        src.write_bytes(b"fake")

        call_count = 0

        def _selective_side_effect(*args, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal call_count
            call_count += 1
            cmd_args = args[0] if args else kwargs.get("args", [])
            if cmd_args:
                candidate = cmd_args[-1]
                if not candidate.startswith("-"):
                    if call_count == 1:
                        # Primary (loudnorm+deess): do NOT create output → trigger fallback
                        return _ok_result()
                    # Fallback (loudnorm only): create output
                    from pathlib import Path as _Path

                    _Path(candidate).parent.mkdir(parents=True, exist_ok=True)
                    _Path(candidate).write_bytes(b"fallback-audio")
            return _ok_result()

        with patch(
            "yt_dbl.utils.audio_processing.run_ffmpeg",
            side_effect=_selective_side_effect,
        ) as mock_ff:
            postprocess_segment(src, dst)
            assert mock_ff.call_count == 2  # failed primary + fallback
            assert dst.exists()
            # Verify the fallback call does NOT contain deess filters
            fallback_args = mock_ff.call_args_list[1][0][0]
            filter_str = " ".join(fallback_args)
            assert "loudnorm" in filter_str
            assert "highshelf" not in filter_str
            assert "compand" not in filter_str
