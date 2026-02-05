"""Root conftest — shared fixtures and CLI hooks for test pyramid."""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

import pytest

from yt_dbl.config import Settings
from yt_dbl.schemas import (
    PipelineState,
    StepName,
    StepStatus,
    VideoMeta,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


# ── CLI hooks ───────────────────────────────────────────────────────────────


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register custom CLI flags."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run tests marked @pytest.mark.slow (network/E2E)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip slow tests unless --run-slow is passed."""
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="need --run-slow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


# ── Shared fixtures ────────────────────────────────────────────────────────

# Short Creative-Commons video for E2E tests (~19 s)
SHORT_VIDEO_URL = "https://www.youtube.com/watch?v=jNQXAC9IVRw"
SHORT_VIDEO_ID = "jNQXAC9IVRw"


@pytest.fixture
def work_dir(tmp_path: Path) -> Path:
    """Provide an isolated temporary work dir for a single test."""
    d = tmp_path / "work"
    d.mkdir()
    return d


@pytest.fixture
def make_settings(work_dir: Path) -> Callable[..., Settings]:
    """Factory fixture: create Settings bound to tmp work_dir."""

    def _make(**overrides: object) -> Settings:
        return Settings(work_dir=work_dir, **overrides)  # type: ignore[arg-type]

    return _make


@pytest.fixture
def settings(make_settings: Callable[..., Settings]) -> Settings:
    """Pre-built Settings pointing to tmp work_dir."""
    return make_settings()


@pytest.fixture
def pipeline_state() -> PipelineState:
    """Minimal fresh PipelineState."""
    return PipelineState(
        video_id="test123",
        url="https://youtube.com/watch?v=test123",
        target_language="ru",
    )


def prefill_download(state: PipelineState, cfg: Settings) -> PipelineState:
    """Utility: populate download step so later steps can run.

    Not a fixture — call directly when a test needs post-download state.
    """
    state.meta = VideoMeta(
        video_id=state.video_id,
        title="Test Video",
        channel="Test Channel",
        duration=60.0,
        url=state.url,
    )
    step = state.get_step(StepName.DOWNLOAD)
    step.status = StepStatus.COMPLETED
    step.outputs = {"video": "video.mp4", "audio": "audio.wav"}

    step_dir = cfg.step_dir(state.video_id, "01_download")
    (step_dir / "video.mp4").write_bytes(b"fake-video")
    (step_dir / "audio.wav").write_bytes(b"fake-audio")
    return state


def prefill_separate(state: PipelineState, cfg: Settings) -> PipelineState:
    """Utility: populate separate step so transcribe+ steps can run.

    Call after prefill_download when tests need post-separation state.
    """
    step = state.get_step(StepName.SEPARATE)
    step.status = StepStatus.COMPLETED
    step.outputs = {"vocals": "vocals.wav", "background": "background.wav"}

    step_dir = cfg.step_dir(state.video_id, "02_separate")
    (step_dir / "vocals.wav").write_bytes(b"fake-vocals")
    (step_dir / "background.wav").write_bytes(b"fake-background")
    return state


# ── E2E fixtures (only used when --run-slow) ───────────────────────────────

# Shared cache so the same video is not re-downloaded across E2E tests inside
# one pytest session.  We use a session-scoped tmp dir and cache there.

_E2E_CACHE_DIR: Path | None = None


@pytest.fixture(scope="session")
def e2e_cache_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Session-wide cache dir for expensive E2E artefacts."""
    global _E2E_CACHE_DIR
    if _E2E_CACHE_DIR is None:
        _E2E_CACHE_DIR = tmp_path_factory.mktemp("e2e_cache")
    return _E2E_CACHE_DIR


@pytest.fixture
def e2e_work_dir(tmp_path: Path) -> Path:
    """Per-test isolated work dir for E2E tests."""
    d = tmp_path / "e2e_work"
    d.mkdir()
    return d


@pytest.fixture
def e2e_settings(e2e_work_dir: Path) -> Settings:
    """Settings configured for E2E tests."""
    return Settings(work_dir=e2e_work_dir)


@pytest.fixture(scope="session")
def ffmpeg_available() -> bool:
    """Check if ffmpeg is available on this machine."""
    return shutil.which("ffmpeg") is not None


@pytest.fixture(scope="session")
def ytdlp_available() -> bool:
    """Check if yt-dlp is available on this machine."""
    return shutil.which("yt-dlp") is not None
