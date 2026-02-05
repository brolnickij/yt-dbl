"""Step 1: Download video and audio from YouTube."""

from __future__ import annotations

import time

from yt_dbl.pipeline.base import PipelineStep
from yt_dbl.schemas import PipelineState, StepName, VideoMeta
from yt_dbl.utils.logging import log_info


class DownloadStep(PipelineStep):
    name = StepName.DOWNLOAD
    description = "Download video + extract audio"

    def run(self, state: PipelineState) -> PipelineState:
        # STUB: simulate download
        log_info("[stub] Simulating video download...")
        time.sleep(0.3)

        state.meta = VideoMeta(
            video_id=state.video_id,
            title="[stub] Sample Video Title",
            channel="[stub] Sample Channel",
            duration=120.0,
            url=state.url,
        )

        result = state.get_step(self.name)
        result.outputs = {
            "video": "video.mp4",
            "audio": "audio.wav",
        }

        log_info("[stub] Download complete (simulated)")
        return state
