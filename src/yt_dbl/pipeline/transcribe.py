"""Step 3: Transcribe speech + diarize speakers."""

from __future__ import annotations

import time

from yt_dbl.pipeline.base import PipelineStep
from yt_dbl.schemas import PipelineState, Segment, Speaker, StepName, Word
from yt_dbl.utils.logging import log_info


class TranscribeStep(PipelineStep):
    name = StepName.TRANSCRIBE
    description = "Transcribe + diarize speakers"

    def validate_inputs(self, state: PipelineState) -> None:
        sep = state.get_step(StepName.SEPARATE)
        if "vocals" not in sep.outputs:
            raise ValueError("No vocals file from separation step")

    def run(self, state: PipelineState) -> PipelineState:
        # STUB: simulate transcription with mock data
        log_info("[stub] Simulating transcription...")
        time.sleep(0.3)

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
                text="Today we will talk about something interesting.",
                start=4.0,
                end=8.0,
                speaker="SPEAKER_00",
                language="en",
                words=[
                    Word(text="Today", start=4.0, end=4.4),
                    Word(text="we", start=4.5, end=4.6),
                    Word(text="will", start=4.7, end=4.9),
                    Word(text="talk", start=5.0, end=5.3),
                    Word(text="about", start=5.4, end=5.7),
                    Word(text="something", start=5.8, end=6.3),
                    Word(text="interesting.", start=6.4, end=7.0),
                ],
            ),
            Segment(
                id=2,
                text="That sounds great, let's get started!",
                start=8.5,
                end=11.0,
                speaker="SPEAKER_01",
                language="en",
                words=[
                    Word(text="That", start=8.5, end=8.8),
                    Word(text="sounds", start=8.9, end=9.2),
                    Word(text="great,", start=9.3, end=9.6),
                    Word(text="let's", start=9.7, end=10.0),
                    Word(text="get", start=10.1, end=10.3),
                    Word(text="started!", start=10.4, end=10.8),
                ],
            ),
        ]

        state.speakers = [
            Speaker(id="SPEAKER_00", total_duration=7.5),
            Speaker(id="SPEAKER_01", total_duration=2.5),
        ]

        result = state.get_step(self.name)
        result.outputs = {"segments": "segments.json"}

        log_info(
            f"[stub] Transcription complete: {len(state.segments)} segments, "
            f"{len(state.speakers)} speakers"
        )
        return state
