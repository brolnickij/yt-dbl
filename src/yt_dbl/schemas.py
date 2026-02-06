"""Data models for the dubbing pipeline."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

__all__ = [
    "STEP_DIRS",
    "STEP_ORDER",
    "PipelineState",
    "Segment",
    "Speaker",
    "StepName",
    "StepResult",
    "StepStatus",
    "VideoMeta",
    "Word",
]

# ── Enums ───────────────────────────────────────────────────────────────────


class StepName(StrEnum):
    """Pipeline step identifiers."""

    DOWNLOAD = "download"
    SEPARATE = "separate"
    TRANSCRIBE = "transcribe"
    TRANSLATE = "translate"
    SYNTHESIZE = "synthesize"
    ASSEMBLE = "assemble"


class StepStatus(StrEnum):
    """Execution status of a pipeline step."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# ── Step metadata ───────────────────────────────────────────────────────────

STEP_ORDER: list[StepName] = [
    StepName.DOWNLOAD,
    StepName.SEPARATE,
    StepName.TRANSCRIBE,
    StepName.TRANSLATE,
    StepName.SYNTHESIZE,
    StepName.ASSEMBLE,
]

STEP_DIRS: dict[StepName, str] = {
    StepName.DOWNLOAD: "01_download",
    StepName.SEPARATE: "02_separate",
    StepName.TRANSCRIBE: "03_transcribe",
    StepName.TRANSLATE: "04_translate",
    StepName.SYNTHESIZE: "05_synthesize",
    StepName.ASSEMBLE: "06_assemble",
}


# ── Word ────────────────────────────────────────────────────────────────────


class Word(BaseModel):
    """A single word with timing information."""

    text: str
    start: float
    end: float
    confidence: float = 1.0


# ── Segment ─────────────────────────────────────────────────────────────────


class Segment(BaseModel):
    """A speech segment with speaker, timing, and word-level detail."""

    id: int
    text: str
    start: float
    end: float
    speaker: str = "SPEAKER_00"
    language: str = ""
    words: list[Word] = Field(default_factory=list)

    # Translation fields (filled after translate step)
    translated_text: str = ""

    # Synthesis fields (filled after synthesize step)
    synth_path: str = ""  # relative path to synthesized WAV
    synth_speed_factor: float = 1.0  # applied speed adjustment

    @property
    def duration(self) -> float:
        return self.end - self.start


# ── Speaker ─────────────────────────────────────────────────────────────────


class Speaker(BaseModel):
    """Speaker with voice reference metadata."""

    id: str  # e.g. "SPEAKER_00"
    reference_start: float = 0.0
    reference_end: float = 0.0
    reference_path: str = ""
    total_duration: float = 0.0  # total speaking time


# ── Video metadata ──────────────────────────────────────────────────────────


class VideoMeta(BaseModel):
    """Metadata extracted from YouTube."""

    video_id: str
    title: str = ""
    channel: str = ""
    duration: float = 0.0
    url: str = ""


# ── Step result ─────────────────────────────────────────────────────────────


class StepResult(BaseModel):
    """Result of a single pipeline step, persisted as checkpoint."""

    step: StepName
    status: StepStatus = StepStatus.PENDING
    started_at: str = ""
    finished_at: str = ""
    duration_sec: float = 0.0
    error: str = ""
    outputs: dict[str, str] = Field(default_factory=dict)  # name → relative path


# ── Pipeline state ──────────────────────────────────────────────────────────


class PipelineState(BaseModel):
    """Full state of a dubbing job, persisted as dubbed/{video_id}/state.json."""

    video_id: str
    url: str = ""
    target_language: str = "ru"
    source_language: str = ""  # auto-detected after transcription
    meta: VideoMeta | None = None
    segments: list[Segment] = Field(default_factory=list)
    speakers: list[Speaker] = Field(default_factory=list)
    steps: dict[StepName, StepResult] = Field(default_factory=dict)

    def get_step(self, name: StepName) -> StepResult:
        if name not in self.steps:
            self.steps[name] = StepResult(step=name)
        return self.steps[name]

    @property
    def last_completed_step(self) -> StepName | None:
        for step_name in reversed(STEP_ORDER):
            result = self.steps.get(step_name)
            if result and result.status == StepStatus.COMPLETED:
                return step_name
        return None

    @property
    def next_step(self) -> StepName | None:
        for step_name in STEP_ORDER:
            result = self.steps.get(step_name)
            if not result or result.status != StepStatus.COMPLETED:
                return step_name
        return None
