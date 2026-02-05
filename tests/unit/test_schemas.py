"""Tests for yt_dbl.schemas — data models and pipeline state."""

import pytest

from yt_dbl.schemas import (
    STEP_DIRS,
    STEP_ORDER,
    PipelineState,
    Segment,
    Speaker,
    StepName,
    StepResult,
    StepStatus,
    VideoMeta,
    Word,
)


class TestWord:
    def test_creation(self) -> None:
        w = Word(text="hello", start=0.0, end=0.5)
        assert w.text == "hello"
        assert w.confidence == 1.0

    def test_custom_confidence(self) -> None:
        w = Word(text="um", start=1.0, end=1.2, confidence=0.42)
        assert w.confidence == pytest.approx(0.42)

    def test_zero_length_word(self) -> None:
        w = Word(text="", start=0.0, end=0.0)
        assert w.text == ""
        assert w.end - w.start == 0.0


class TestSegment:
    def test_duration(self) -> None:
        seg = Segment(id=0, text="test", start=1.0, end=3.5)
        assert seg.duration == 2.5

    def test_defaults(self) -> None:
        seg = Segment(id=0, text="test", start=0.0, end=1.0)
        assert seg.speaker == "SPEAKER_00"
        assert seg.words == []
        assert seg.translated_text == ""

    def test_zero_duration(self) -> None:
        seg = Segment(id=0, text="silence", start=5.0, end=5.0)
        assert seg.duration == 0.0

    def test_with_words(self) -> None:
        words = [
            Word(text="hello", start=0.0, end=0.3),
            Word(text="world", start=0.4, end=0.8),
        ]
        seg = Segment(id=1, text="hello world", start=0.0, end=0.8, words=words)
        assert len(seg.words) == 2
        assert seg.words[0].text == "hello"

    def test_translated_text_field(self) -> None:
        seg = Segment(id=0, text="Hello", start=0.0, end=1.0, translated_text="Привет")
        assert seg.translated_text == "Привет"


class TestSpeaker:
    def test_defaults(self) -> None:
        sp = Speaker(id="SPEAKER_01")
        assert sp.reference_start == 0.0
        assert sp.reference_end == 0.0
        assert sp.reference_path == ""
        assert sp.total_duration == 0.0

    def test_with_reference(self) -> None:
        sp = Speaker(
            id="SPEAKER_00",
            reference_start=10.0,
            reference_end=17.0,
            reference_path="ref.wav",
            total_duration=120.5,
        )
        assert sp.reference_end - sp.reference_start == 7.0
        assert sp.reference_path == "ref.wav"


class TestVideoMeta:
    def test_minimal(self) -> None:
        m = VideoMeta(video_id="abc")
        assert m.title == ""
        assert m.duration == 0.0

    def test_full(self) -> None:
        m = VideoMeta(
            video_id="abc",
            title="Test",
            channel="Ch",
            duration=300.0,
            url="https://youtube.com/watch?v=abc",
        )
        assert m.channel == "Ch"
        assert m.duration == 300.0


class TestStepResult:
    def test_defaults(self) -> None:
        r = StepResult(step=StepName.DOWNLOAD)
        assert r.status == StepStatus.PENDING
        assert r.started_at == ""
        assert r.finished_at == ""
        assert r.duration_sec == 0.0
        assert r.error == ""
        assert r.outputs == {}

    def test_failed_step(self) -> None:
        r = StepResult(
            step=StepName.TRANSCRIBE,
            status=StepStatus.FAILED,
            error="Model OOM",
        )
        assert r.status == StepStatus.FAILED
        assert "OOM" in r.error


class TestStepEnums:
    def test_step_order_length(self) -> None:
        assert len(STEP_ORDER) == 6

    def test_step_dirs_keys_match_order(self) -> None:
        assert list(STEP_DIRS.keys()) == STEP_ORDER

    def test_step_dirs_prefixes(self) -> None:
        for i, (_name, dirname) in enumerate(STEP_DIRS.items(), start=1):
            assert dirname.startswith(f"{i:02d}_")

    def test_status_values(self) -> None:
        assert StepStatus.PENDING.value == "pending"
        assert StepStatus.RUNNING.value == "running"
        assert StepStatus.COMPLETED.value == "completed"
        assert StepStatus.FAILED.value == "failed"


class TestPipelineState:
    def test_next_step_fresh(self) -> None:
        state = PipelineState(video_id="abc123")
        assert state.next_step == StepName.DOWNLOAD

    def test_next_step_after_download(self) -> None:
        state = PipelineState(video_id="abc123")
        step = state.get_step(StepName.DOWNLOAD)
        step.status = StepStatus.COMPLETED
        assert state.next_step == StepName.SEPARATE

    def test_last_completed_step_none(self) -> None:
        state = PipelineState(video_id="abc123")
        assert state.last_completed_step is None

    def test_last_completed_step(self) -> None:
        state = PipelineState(video_id="abc123")
        for name in [StepName.DOWNLOAD, StepName.SEPARATE, StepName.TRANSCRIBE]:
            state.get_step(name).status = StepStatus.COMPLETED
        assert state.last_completed_step == StepName.TRANSCRIBE

    def test_all_steps_done(self) -> None:
        state = PipelineState(video_id="abc123")
        for name in STEP_ORDER:
            state.get_step(name).status = StepStatus.COMPLETED
        assert state.next_step is None

    def test_get_step_creates_on_demand(self) -> None:
        state = PipelineState(video_id="abc123")
        result = state.get_step(StepName.TRANSLATE)
        assert result.step == StepName.TRANSLATE
        assert result.status == StepStatus.PENDING

    def test_get_step_idempotent(self) -> None:
        """Calling get_step twice returns the same object."""
        state = PipelineState(video_id="abc123")
        r1 = state.get_step(StepName.DOWNLOAD)
        r1.status = StepStatus.RUNNING
        r2 = state.get_step(StepName.DOWNLOAD)
        assert r2.status == StepStatus.RUNNING
        assert r1 is r2

    def test_failed_step_blocks_next(self) -> None:
        """A failed step still appears as next_step (not completed)."""
        state = PipelineState(video_id="abc123")
        state.get_step(StepName.DOWNLOAD).status = StepStatus.COMPLETED
        state.get_step(StepName.SEPARATE).status = StepStatus.FAILED
        assert state.next_step == StepName.SEPARATE

    def test_default_target_language(self) -> None:
        state = PipelineState(video_id="abc")
        assert state.target_language == "ru"

    def test_serialization_roundtrip(self) -> None:
        state = PipelineState(
            video_id="abc123",
            url="https://youtube.com/watch?v=abc123",
            target_language="ru",
            meta=VideoMeta(video_id="abc123", title="Test", channel="Ch", duration=60.0),
            segments=[
                Segment(
                    id=0,
                    text="Hello",
                    start=0.0,
                    end=1.0,
                    words=[Word(text="Hello", start=0.0, end=0.5)],
                )
            ],
            speakers=[Speaker(id="SPEAKER_00", total_duration=10.0)],
        )
        state.get_step(StepName.DOWNLOAD).status = StepStatus.COMPLETED

        json_str = state.model_dump_json()
        restored = PipelineState.model_validate_json(json_str)

        assert restored.video_id == "abc123"
        assert restored.meta is not None
        assert restored.meta.title == "Test"
        assert len(restored.segments) == 1
        assert len(restored.segments[0].words) == 1
        assert restored.get_step(StepName.DOWNLOAD).status == StepStatus.COMPLETED

    def test_roundtrip_preserves_step_outputs(self) -> None:
        state = PipelineState(video_id="abc")
        step = state.get_step(StepName.DOWNLOAD)
        step.status = StepStatus.COMPLETED
        step.outputs = {"video": "video.mp4", "audio": "audio.wav"}

        restored = PipelineState.model_validate_json(state.model_dump_json())
        assert restored.get_step(StepName.DOWNLOAD).outputs["video"] == "video.mp4"
