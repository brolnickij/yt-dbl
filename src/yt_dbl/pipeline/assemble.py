"""Step 6: Assemble final video with dubbed audio."""

from __future__ import annotations

import time

from yt_dbl.pipeline.base import PipelineStep
from yt_dbl.schemas import PipelineState, StepName
from yt_dbl.utils.logging import log_info


class AssembleStep(PipelineStep):
    name = StepName.ASSEMBLE
    description = "Assemble final video"

    def validate_inputs(self, state: PipelineState) -> None:
        synth = state.get_step(StepName.SYNTHESIZE)
        if not synth.outputs:
            raise ValueError("No synthesized segments")

    def run(self, state: PipelineState) -> PipelineState:
        # STUB: simulate assembly
        log_info("[stub] Simulating final assembly...")
        time.sleep(0.3)

        output_name = f"result.{self.settings.output_format}"

        result = state.get_step(self.name)
        result.outputs = {
            "result": output_name,
            "subtitles": "subtitles.srt",
        }

        log_info(f"[stub] Assembly complete → {output_name}")
        return state
