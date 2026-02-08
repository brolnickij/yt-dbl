"""Tests for yt_dbl.models.registry — model info, download status, sizing."""

from __future__ import annotations

from pathlib import Path

import pytest

from yt_dbl.models.registry import (
    MODEL_REGISTRY,
    _hf_cache_dir,
    _repo_dir_name,
    check_model_downloaded,
    check_separator_downloaded,
    download_model,
    format_model_size,
    get_model_size,
)


class TestModelRegistry:
    def test_registry_not_empty(self) -> None:
        assert len(MODEL_REGISTRY) >= 3

    def test_registry_entries_have_fields(self) -> None:
        for info in MODEL_REGISTRY:
            assert info.repo_id
            assert info.purpose
            assert info.step
            assert info.approx_size

    def test_registry_includes_core_models(self) -> None:
        repo_ids = {m.repo_id for m in MODEL_REGISTRY}
        assert "mlx-community/VibeVoice-ASR-4bit" in repo_ids
        assert "mlx-community/Qwen3-ForcedAligner-0.6B-8bit" in repo_ids
        assert "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16" in repo_ids


class TestRepoDirName:
    def test_standard(self) -> None:
        assert _repo_dir_name("mlx-community/VibeVoice-ASR-4bit") == (
            "models--mlx-community--VibeVoice-ASR-4bit"
        )

    def test_simple(self) -> None:
        assert _repo_dir_name("org/model") == "models--org--model"


class TestCheckDownloaded:
    def test_not_downloaded(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HF_HOME", str(tmp_path / "hf_home"))
        assert check_model_downloaded("nonexistent/model") is False

    def test_downloaded(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        hf_dir = tmp_path / "hf_home" / "hub"
        repo_dir = hf_dir / "models--test--model" / "snapshots" / "abc123"
        repo_dir.mkdir(parents=True)
        (repo_dir / "config.json").write_text("{}")

        monkeypatch.setenv("HF_HOME", str(tmp_path / "hf_home"))
        assert check_model_downloaded("test/model") is True

    def test_empty_snapshots(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        hf_dir = tmp_path / "hf_home" / "hub"
        snapshots = hf_dir / "models--test--model" / "snapshots"
        snapshots.mkdir(parents=True)  # exists but empty

        monkeypatch.setenv("HF_HOME", str(tmp_path / "hf_home"))
        assert check_model_downloaded("test/model") is False


class TestGetModelSize:
    def test_not_downloaded(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HF_HOME", str(tmp_path / "hf_home"))
        assert get_model_size("nonexistent/model") == 0

    def test_with_blobs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        hf_dir = tmp_path / "hf_home" / "hub"
        blobs = hf_dir / "models--test--model" / "blobs"
        blobs.mkdir(parents=True)
        (blobs / "blob1").write_bytes(b"x" * 1000)
        (blobs / "blob2").write_bytes(b"y" * 2000)

        monkeypatch.setenv("HF_HOME", str(tmp_path / "hf_home"))
        assert get_model_size("test/model") == 3000


class TestSeparatorCheck:
    def test_not_downloaded(self, tmp_path: Path) -> None:
        assert check_separator_downloaded(tmp_path) is False

    def test_downloaded(self, tmp_path: Path) -> None:
        (tmp_path / "model_bs_roformer_ep_317_sdr_12.9755.ckpt").write_bytes(b"fake")
        assert check_separator_downloaded(tmp_path) is True


class TestFormatSize:
    @pytest.mark.parametrize(
        ("size", "expected"),
        [
            (0, "—"),
            (500 * 1024 * 1024, "500 MB"),
            (int(1.5 * 1024**3), "1.5 GB"),
        ],
        ids=["zero", "megabytes", "gigabytes"],
    )
    def test_format(self, size: int, expected: str) -> None:
        assert expected in format_model_size(size)


class TestHFCacheDir:
    def test_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HF_HOME", raising=False)
        monkeypatch.delenv("HF_HUB_CACHE", raising=False)
        result = _hf_cache_dir()
        assert result == Path.home() / ".cache" / "huggingface" / "hub"

    def test_hf_home(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HF_HOME", str(tmp_path / "custom"))
        monkeypatch.delenv("HF_HUB_CACHE", raising=False)
        result = _hf_cache_dir()
        assert result == tmp_path / "custom" / "hub"

    def test_hf_hub_cache(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HF_HOME", raising=False)
        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub_cache"))
        result = _hf_cache_dir()
        assert result == tmp_path / "hub_cache"


class TestDownloadModel:
    def test_passes_token(self) -> None:
        """download_model forwards token to snapshot_download."""
        from unittest.mock import patch

        with patch("huggingface_hub.snapshot_download", return_value="ok") as mock_dl:
            download_model("org/model", token="hf_secret")

        mock_dl.assert_called_once_with("org/model", token="hf_secret")

    def test_empty_token_passes_none(self) -> None:
        """Empty token string is converted to None for snapshot_download."""
        from unittest.mock import patch

        with patch("huggingface_hub.snapshot_download", return_value="ok") as mock_dl:
            download_model("org/model")

        mock_dl.assert_called_once_with("org/model", token=None)
