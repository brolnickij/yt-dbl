"""Tests for CLI commands."""

from typer.testing import CliRunner

from yt_dbl.cli import app

runner = CliRunner()


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
        assert "--target-language" in result.output
        assert "--from-step" in result.output

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
        result = runner.invoke(app, ["models", "download"])
        assert result.exit_code == 0
