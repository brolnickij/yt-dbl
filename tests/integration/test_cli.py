"""Tests for CLI commands."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from yt_dbl.cli import _extract_video_id, _step_name_from_str, app
from yt_dbl.schemas import PipelineState, StepName, StepStatus

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    """Strip ANSI escape codes so assertions survive Rich markup."""
    return _ANSI_RE.sub("", text)


class TestCLI:
    def test_version(self) -> None:
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_help(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "dub" in result.output
        assert "resume" in result.output
        assert "models" in result.output

    def test_dub_help(self) -> None:
        result = runner.invoke(app, ["dub", "--help"])
        assert result.exit_code == 0
        output = _plain(result.output)
        assert "--target-language" in output
        assert "--from-step" in output

    def test_resume_nonexistent(self) -> None:
        result = runner.invoke(app, ["resume", "nonexistent_video_id_xyz"])
        assert result.exit_code == 1

    def test_status_nonexistent(self) -> None:
        result = runner.invoke(app, ["status", "nonexistent_video_id_xyz"])
        assert result.exit_code == 1

    def test_models_list(self) -> None:
        result = runner.invoke(app, ["models", "list"])
        assert result.exit_code == 0

    def test_models_download(self) -> None:
        with patch(
            "yt_dbl.models.registry.check_model_downloaded",
            return_value=True,
        ):
            result = runner.invoke(app, ["models", "download"])
        assert result.exit_code == 0


class TestExtractVideoId:
    """Unit tests for _extract_video_id helper."""

    def test_standard_url(self) -> None:
        assert _extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_short_url(self) -> None:
        assert _extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_shorts_url(self) -> None:
        assert _extract_video_id("https://youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_embed_url(self) -> None:
        assert _extract_video_id("https://youtube.com/v/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_url_with_extra_params(self) -> None:
        vid = _extract_video_id("https://www.youtube.com/watch?v=abcdefghijk&list=PL123")
        assert vid == "abcdefghijk"

    def test_fallback_sanitizes(self) -> None:
        """Non-YouTube URL should be sanitized as fallback."""
        vid = _extract_video_id("https://example.com/some-video")
        assert len(vid) <= 64
        assert all(c.isalnum() or c in "-_" for c in vid)


class TestStepNameFromStr:
    def test_valid_step(self) -> None:
        assert _step_name_from_str("download") == StepName.DOWNLOAD
        assert _step_name_from_str("TRANSLATE") == StepName.TRANSLATE

    def test_invalid_step(self) -> None:
        import pytest
        import typer

        with pytest.raises(typer.BadParameter, match="nonexistent"):
            _step_name_from_str("nonexistent")


class TestDubCommand:
    """Integration tests for the `dub` command (mocked pipeline)."""

    @patch("yt_dbl.pipeline.runner.PipelineRunner")
    @patch("yt_dbl.pipeline.runner.save_state")
    @patch("yt_dbl.pipeline.runner.load_state", return_value=None)
    def test_dub_new_job(
        self,
        mock_load: MagicMock,
        mock_save: MagicMock,
        mock_runner_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_runner = MagicMock()
        mock_runner_cls.return_value = mock_runner

        result = runner.invoke(app, ["dub", "https://youtube.com/watch?v=dQw4w9WgXcQ"])

        assert result.exit_code == 0
        mock_runner.run.assert_called_once()

    @patch("yt_dbl.pipeline.runner.PipelineRunner")
    @patch("yt_dbl.pipeline.runner.save_state")
    @patch("yt_dbl.pipeline.runner.load_state")
    def test_dub_resumes_existing(
        self,
        mock_load: MagicMock,
        mock_save: MagicMock,
        mock_runner_cls: MagicMock,
    ) -> None:
        existing_state = PipelineState(
            video_id="dQw4w9WgXcQ",
            url="https://youtube.com/watch?v=dQw4w9WgXcQ",
        )
        existing_state.get_step(StepName.DOWNLOAD).status = StepStatus.COMPLETED
        mock_load.return_value = existing_state

        mock_runner = MagicMock()
        mock_runner_cls.return_value = mock_runner

        result = runner.invoke(app, ["dub", "https://youtube.com/watch?v=dQw4w9WgXcQ"])

        assert result.exit_code == 0
        # Should NOT create a new state — load returned existing
        mock_save.assert_not_called()

    @patch("yt_dbl.pipeline.runner.PipelineRunner")
    @patch("yt_dbl.pipeline.runner.save_state")
    @patch("yt_dbl.pipeline.runner.load_state", return_value=None)
    def test_dub_with_from_step(
        self,
        mock_load: MagicMock,
        mock_save: MagicMock,
        mock_runner_cls: MagicMock,
    ) -> None:
        mock_runner = MagicMock()
        mock_runner_cls.return_value = mock_runner

        result = runner.invoke(
            app,
            ["dub", "https://youtube.com/watch?v=abc12345678", "--from-step", "translate"],
        )

        assert result.exit_code == 0
        call_kwargs = mock_runner.run.call_args
        assert call_kwargs[1]["from_step"] == StepName.TRANSLATE


class TestResumeCommand:
    @patch("yt_dbl.pipeline.runner.PipelineRunner")
    @patch("yt_dbl.pipeline.runner.load_state")
    def test_resume_existing_job(
        self,
        mock_load: MagicMock,
        mock_runner_cls: MagicMock,
    ) -> None:
        state = PipelineState(video_id="test123", url="https://example.com")
        state.get_step(StepName.DOWNLOAD).status = StepStatus.COMPLETED
        mock_load.return_value = state

        mock_runner = MagicMock()
        mock_runner_cls.return_value = mock_runner

        result = runner.invoke(app, ["resume", "test123"])

        assert result.exit_code == 0
        mock_runner.run.assert_called_once()

    @patch("yt_dbl.pipeline.runner.load_state")
    def test_resume_completed_job(self, mock_load: MagicMock) -> None:
        state = PipelineState(video_id="test123", url="https://example.com")
        for step in StepName:
            state.get_step(step).status = StepStatus.COMPLETED
        mock_load.return_value = state

        result = runner.invoke(app, ["resume", "test123"])
        assert result.exit_code == 0
        assert "completed" in result.output.lower()


class TestStatusCommand:
    @patch("yt_dbl.cli.settings")
    def test_status_shows_table(self, mock_settings: MagicMock, tmp_path: Path) -> None:
        from yt_dbl.schemas import VideoMeta

        state = PipelineState(
            video_id="test123",
            url="https://example.com",
            meta=VideoMeta(video_id="test123", title="Test Title", channel="Ch", duration=60.0),
        )
        state.get_step(StepName.DOWNLOAD).status = StepStatus.COMPLETED
        state.get_step(StepName.DOWNLOAD).duration_sec = 5.3
        state.get_step(StepName.SEPARATE).status = StepStatus.FAILED
        state.get_step(StepName.SEPARATE).error = "Some error happened during separation"

        # Write a state file
        job_dir = tmp_path / "test123"
        job_dir.mkdir(parents=True)
        state_file = job_dir / "state.json"
        state_file.write_text(state.model_dump_json(indent=2))

        mock_settings.job_dir.return_value = job_dir
        # Make load_state find it
        with patch("yt_dbl.pipeline.runner.load_state", return_value=state):
            result = runner.invoke(app, ["status", "test123"])

        assert result.exit_code == 0
        assert "completed" in result.output.lower()
        assert "failed" in result.output.lower()
        assert "Test Title" in result.output
