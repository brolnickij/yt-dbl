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


def prefill_translate(state: PipelineState, cfg: Settings) -> PipelineState:
    """Utility: populate translate step so synthesize+ steps can run.

    Call after prefill_transcribe when tests need post-translation state.
    """
    import json as _json

    for seg in state.segments:
        seg.translated_text = f"[translated] {seg.text}"

    step = state.get_step(StepName.TRANSLATE)
    step.status = StepStatus.COMPLETED
    step.outputs = {"translations": "translations.json", "subtitles": "subtitles.srt"}

    step_dir = cfg.step_dir(state.video_id, "04_translate")
    data = [{"id": seg.id, "translated_text": seg.translated_text} for seg in state.segments]
    (step_dir / "translations.json").write_text(
        _json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return state


def prefill_synthesize(state: PipelineState, cfg: Settings) -> PipelineState:
    """Utility: populate synthesize step so assemble+ steps can run.

    Call after prefill_translate when tests need post-synthesis state.
    """
    import json as _json

    for seg in state.segments:
        seg.synth_path = f"segment_{seg.id:04d}.wav"
        seg.synth_speed_factor = 1.0

    step = state.get_step(StepName.SYNTHESIZE)
    step.status = StepStatus.COMPLETED
    step.outputs = {f"seg_{seg.id}": seg.synth_path for seg in state.segments}
    step.outputs["meta"] = "synth_meta.json"

    step_dir = cfg.step_dir(state.video_id, "05_synthesize")
    for seg in state.segments:
        (step_dir / seg.synth_path).write_bytes(b"fake-synth")

    meta = {
        "segments": [
            {
                "id": seg.id,
                "synth_path": seg.synth_path,
                "synth_speed_factor": seg.synth_speed_factor,
            }
            for seg in state.segments
        ],
        "speakers": [{"id": s.id, "reference_path": s.reference_path} for s in state.speakers],
    }
    (step_dir / "synth_meta.json").write_text(_json.dumps(meta, indent=2), encoding="utf-8")
    return state


def prefill_transcribe(state: PipelineState, cfg: Settings) -> PipelineState:
    """Utility: populate transcribe step so translate+ steps can run.

    Call after prefill_separate when tests need post-transcription state.
    """
    from yt_dbl.schemas import Segment, Speaker, Word

    state.segments = [
        Segment(
            id=0,
            text="Hello, welcome to this video.",
            start=0.0,
            end=3.5,
            speaker="SPEAKER_00",
            language="en",
            words=[
                Word(text="Hello,", start=0.0, end=0.5),
                Word(text="welcome", start=0.6, end=1.0),
                Word(text="to", start=1.1, end=1.2),
                Word(text="this", start=1.3, end=1.5),
                Word(text="video.", start=1.6, end=2.0),
            ],
        ),
        Segment(
            id=1,
            text="That sounds great!",
            start=4.0,
            end=6.0,
            speaker="SPEAKER_01",
            language="en",
            words=[
                Word(text="That", start=4.0, end=4.3),
                Word(text="sounds", start=4.4, end=4.8),
                Word(text="great!", start=4.9, end=5.5),
            ],
        ),
    ]
    state.speakers = [
        Speaker(id="SPEAKER_00", total_duration=3.5),
        Speaker(id="SPEAKER_01", total_duration=2.0),
    ]

    step = state.get_step(StepName.TRANSCRIBE)
    step.status = StepStatus.COMPLETED
    step.outputs = {"segments": "segments.json"}

    import json

    step_dir = cfg.step_dir(state.video_id, "03_transcribe")
    data = {
        "segments": [s.model_dump() for s in state.segments],
        "speakers": [s.model_dump() for s in state.speakers],
    }
    (step_dir / "segments.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
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


@pytest.fixture(scope="session")
def audio_separator_available() -> bool:
    """Check if audio-separator is importable."""
    try:
        import audio_separator.separator  # noqa: F401

        return True  # noqa: TRY300
    except ImportError:
        return False
