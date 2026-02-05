"""Step 2: Separate vocals from background using ensemble."""

from __future__ import annotations

import time

from yt_dbl.pipeline.base import PipelineStep
from yt_dbl.schemas import PipelineState, StepName
from yt_dbl.utils.logging import log_info


class SeparateStep(PipelineStep):
    name = StepName.SEPARATE
    description = "Separate vocals from background (ensemble)"

    def validate_inputs(self, state: PipelineState) -> None:
        dl = state.get_step(StepName.DOWNLOAD)
        if "audio" not in dl.outputs:
            raise ValueError("No audio file from download step")

    def run(self, state: PipelineState) -> PipelineState:
        # STUB: simulate separation
        log_info("[stub] Simulating vocal separation...")
        time.sleep(0.3)

        result = state.get_step(self.name)
        result.outputs = {
            "vocals": "vocals.wav",
            "background": "background.wav",
        }

        log_info("[stub] Separation complete (simulated)")
        return state
