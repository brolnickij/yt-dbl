"""Tests for yt_dbl.pipeline.download — YouTube download step (mocked)."""

from __future__ import annotations

import json
import subprocess
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from yt_dbl.config import Settings
from yt_dbl.pipeline.base import DownloadError, StepValidationError
from yt_dbl.pipeline.download import DownloadStep
from yt_dbl.schemas import STEP_DIRS, PipelineState, StepName

if TYPE_CHECKING:
    from pathlib import Path

FAKE_META = {
    "title": "Test Video Title",
    "channel": "Test Channel",
    "uploader": "Test Uploader",
    "duration": 120.5,
    "id": "dQw4w9WgXcQ",
}


class TestDownloadStep:
    def _make_step(self, tmp_path: Path) -> tuple[DownloadStep, Settings, PipelineState]:
        cfg = Settings(work_dir=tmp_path / "work")
        step_dir = cfg.step_dir("test123", STEP_DIRS[StepName.DOWNLOAD])
        step = DownloadStep(settings=cfg, work_dir=step_dir)
        state = PipelineState(
            video_id="test123",
            url="https://youtube.com/watch?v=test123",
        )
        return step, cfg, state

    def test_validate_inputs_no_url(self, tmp_path: Path) -> None:
        step, _, state = self._make_step(tmp_path)
        state.url = ""
        with pytest.raises(StepValidationError, match="No URL"):
            step.validate_inputs(state)

    def test_validate_inputs_ok(self, tmp_path: Path) -> None:
        step, _, state = self._make_step(tmp_path)
        step.validate_inputs(state)  # Should not raise

    @patch("yt_dbl.pipeline.download.DownloadStep._download_video")
    @patch("yt_dbl.pipeline.download.DownloadStep._fetch_metadata")
    @patch("yt_dbl.pipeline.download.extract_audio")
    def test_run_success(
        self,
        mock_extract: MagicMock,
        mock_meta: MagicMock,
        mock_download: MagicMock,
        tmp_path: Path,
    ) -> None:
        step, _cfg, state = self._make_step(tmp_path)

        mock_meta.return_value = FAKE_META

        # Simulate file creation by download and extract
        def create_video(url: str, path: Path) -> None:
            path.write_bytes(b"fake-video-content")

        def create_audio(video_path: Path, audio_path: Path, **kwargs: object) -> Path:
            audio_path.write_bytes(b"fake-audio-content")
            return audio_path

        mock_download.side_effect = create_video
        mock_extract.side_effect = create_audio

        state = step.run(state)

        assert state.meta is not None
        assert state.meta.title == "Test Video Title"
        assert state.meta.channel == "Test Channel"
        assert state.meta.duration == 120.5

        result = state.get_step(StepName.DOWNLOAD)
        assert result.outputs["video"] == "video.mp4"
        assert result.outputs["audio"] == "audio.wav"

    @patch("yt_dbl.pipeline.download.DownloadStep._fetch_metadata")
    def test_fetch_metadata_timeout(
        self,
        mock_meta: MagicMock,
        tmp_path: Path,
    ) -> None:
        step, _, state = self._make_step(tmp_path)
        mock_meta.side_effect = DownloadError("Metadata fetch timed out (30s)")

        with pytest.raises(DownloadError, match="timed out"):
            step.run(state)

    @patch("yt_dbl.pipeline.download.DownloadStep._download_video")
    @patch("yt_dbl.pipeline.download.DownloadStep._fetch_metadata")
    def test_skips_existing_files(
        self,
        mock_meta: MagicMock,
        mock_download: MagicMock,
        tmp_path: Path,
    ) -> None:
        step, _cfg, state = self._make_step(tmp_path)
        mock_meta.return_value = FAKE_META

        # Pre-create files
        (step.step_dir / "video.mp4").write_bytes(b"existing-video")
        (step.step_dir / "audio.wav").write_bytes(b"existing-audio")

        state = step.run(state)

        mock_download.assert_not_called()
        assert state.meta is not None
        assert state.meta.title == "Test Video Title"


class TestFetchMetadata:
    def test_parses_json_output(self, tmp_path: Path) -> None:
        cfg = Settings(work_dir=tmp_path / "work")
        step_dir = cfg.step_dir("test123", STEP_DIRS[StepName.DOWNLOAD])
        step = DownloadStep(settings=cfg, work_dir=step_dir)

        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps(FAKE_META), stderr=""
        )

        with patch("subprocess.run", return_value=mock_result):
            meta = step._fetch_metadata("https://youtube.com/watch?v=test")

        assert meta["title"] == "Test Video Title"
        assert meta["duration"] == 120.5

    def test_raises_on_failure(self, tmp_path: Path) -> None:
        cfg = Settings(work_dir=tmp_path / "work")
        step_dir = cfg.step_dir("test123", STEP_DIRS[StepName.DOWNLOAD])
        step = DownloadStep(settings=cfg, work_dir=step_dir)

        with (
            patch(
                "subprocess.run",
                side_effect=subprocess.CalledProcessError(1, "yt-dlp", stderr="Video unavailable"),
            ),
            pytest.raises(DownloadError, match="Failed to fetch metadata"),
        ):
            step._fetch_metadata("https://youtube.com/watch?v=invalid")


class TestDownloadVideo:
    """Tests for _download_video error handling."""

    @patch("subprocess.Popen")
    def test_yt_dlp_not_found(self, mock_popen: MagicMock, tmp_path: Path) -> None:
        mock_popen.side_effect = FileNotFoundError()
        cfg = Settings(work_dir=tmp_path / "work")
        step_dir = cfg.step_dir("test123", STEP_DIRS[StepName.DOWNLOAD])
        step = DownloadStep(settings=cfg, work_dir=step_dir)

        with pytest.raises(DownloadError, match="yt-dlp not found"):
            step._download_video("https://youtube.com/watch?v=x", step_dir / "video.mp4")

    @patch("subprocess.Popen")
    def test_nonzero_exit_code(self, mock_popen: MagicMock, tmp_path: Path) -> None:
        proc = MagicMock()
        proc.stdout = iter(["[download] 100% done\n"])
        proc.wait.return_value = 1
        mock_popen.return_value = proc

        cfg = Settings(work_dir=tmp_path / "work")
        step_dir = cfg.step_dir("test123", STEP_DIRS[StepName.DOWNLOAD])
        step = DownloadStep(settings=cfg, work_dir=step_dir)

        with pytest.raises(DownloadError, match="exited with code 1"):
            step._download_video("https://youtube.com/watch?v=x", step_dir / "video.mp4")

    @patch("subprocess.Popen")
    def test_kills_process_on_unexpected_error(self, mock_popen: MagicMock, tmp_path: Path) -> None:
        """If an unexpected error occurs while reading stdout, yt-dlp is killed."""
        proc = MagicMock()
        proc.stdout = iter(["boom"])
        # Simulate the progress bar raising an unexpected error
        proc.wait.side_effect = RuntimeError("unexpected")
        mock_popen.return_value = proc

        cfg = Settings(work_dir=tmp_path / "work")
        step_dir = cfg.step_dir("test123", STEP_DIRS[StepName.DOWNLOAD])
        step = DownloadStep(settings=cfg, work_dir=step_dir)

        with pytest.raises(RuntimeError, match="unexpected"):
            step._download_video("https://youtube.com/watch?v=x", step_dir / "video.mp4")

        proc.kill.assert_called_once()


class TestDownloadStepOutputVerification:
    """Verify that run() checks output presence."""

    @patch("yt_dbl.pipeline.download.DownloadStep._download_video")
    @patch("yt_dbl.pipeline.download.DownloadStep._fetch_metadata")
    @patch("yt_dbl.pipeline.download.extract_audio")
    def test_missing_video_raises(
        self,
        mock_extract: MagicMock,
        mock_meta: MagicMock,
        mock_download: MagicMock,
        tmp_path: Path,
    ) -> None:
        cfg = Settings(work_dir=tmp_path / "work")
        step_dir = cfg.step_dir("test123", STEP_DIRS[StepName.DOWNLOAD])
        step = DownloadStep(settings=cfg, work_dir=step_dir)
        state = PipelineState(video_id="test123", url="https://youtube.com/watch?v=test123")

        mock_meta.return_value = FAKE_META
        # download does nothing — video file won't exist
        mock_download.side_effect = lambda *a, **kw: None
        mock_extract.side_effect = lambda *a, **kw: None

        with pytest.raises(DownloadError, match="Video file was not created"):
            step.run(state)
