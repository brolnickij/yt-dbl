"""Step 5: Synthesize translated speech with voice cloning."""

from __future__ import annotations

import time

from yt_dbl.pipeline.base import PipelineStep
from yt_dbl.schemas import PipelineState, StepName
from yt_dbl.utils.logging import log_info


class SynthesizeStep(PipelineStep):
    name = StepName.SYNTHESIZE
    description = "Synthesize speech (voice cloning)"

    def validate_inputs(self, state: PipelineState) -> None:
        if not state.segments:
            raise ValueError("No segments to synthesize")
        if not any(seg.translated_text for seg in state.segments):
            raise ValueError("Segments have no translated text")

    def run(self, state: PipelineState) -> PipelineState:
        # STUB: simulate TTS
        log_info("[stub] Simulating speech synthesis...")
        time.sleep(0.3)

        result = state.get_step(self.name)
        outputs: dict[str, str] = {}

        for seg in state.segments:
            filename = f"segment_{seg.id:04d}.wav"
            outputs[f"seg_{seg.id}"] = filename

        result.outputs = outputs

        log_info(f"[stub] Synthesis complete: {len(state.segments)} segments")
        return state
