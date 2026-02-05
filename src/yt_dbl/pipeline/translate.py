"""Step 4: Translate segments via Claude (3-pass)."""

from __future__ import annotations

import time

from yt_dbl.pipeline.base import PipelineStep
from yt_dbl.schemas import PipelineState, StepName
from yt_dbl.utils.logging import log_info


class TranslateStep(PipelineStep):
    name = StepName.TRANSLATE
    description = "Translate (3-pass via Claude)"

    def validate_inputs(self, state: PipelineState) -> None:
        if not state.segments:
            raise ValueError("No segments to translate")

    def run(self, state: PipelineState) -> PipelineState:
        # STUB: simulate translation
        log_info("[stub] Simulating 3-pass translation...")
        time.sleep(0.3)

        mock_translations = {
            0: "Привет, добро пожаловать в это видео.",
            1: "Сегодня мы поговорим о кое-чём интересном.",  # noqa: RUF001
            2: "Звучит отлично, давайте начнём!",
        }

        for seg in state.segments:
            seg.translated_text = mock_translations.get(seg.id, f"[перевод] {seg.text}")

        result = state.get_step(self.name)
        result.outputs = {
            "subtitles": "subtitles.srt",
        }

        log_info(f"[stub] Translation complete: {len(state.segments)} segments")
        return state
