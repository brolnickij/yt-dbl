"""Tests for yt_dbl.schemas — data models and pipeline state."""

from yt_dbl.schemas import (
    STEP_ORDER,
    PipelineState,
    Segment,
    Speaker,
    StepName,
    StepStatus,
    VideoMeta,
    Word,
)


class TestWord:
    def test_creation(self) -> None:
        w = Word(text="hello", start=0.0, end=0.5)
        assert w.text == "hello"
        assert w.confidence == 1.0


class TestSegment:
    def test_duration(self) -> None:
        seg = Segment(id=0, text="test", start=1.0, end=3.5)
        assert seg.duration == 2.5

    def test_defaults(self) -> None:
        seg = Segment(id=0, text="test", start=0.0, end=1.0)
        assert seg.speaker == "SPEAKER_00"
        assert seg.words == []
        assert seg.translated_text == ""


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
