"""Step 2: Separate vocals from background using BS-RoFormer."""

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import TYPE_CHECKING

from yt_dbl.pipeline.base import PipelineStep, SeparationError, StepValidationError
from yt_dbl.schemas import STEP_DIRS, StepName
from yt_dbl.utils.logging import format_file_size, log_info, log_warning

if TYPE_CHECKING:
    from yt_dbl.schemas import PipelineState

__all__ = ["SeparateStep"]

# Standard output filenames
VOCALS_FILENAME = "vocals.wav"
BACKGROUND_FILENAME = "background.wav"

# Stem labels used by audio-separator
_VOCALS_LABEL = "Vocals"
_INSTRUMENTAL_LABEL = "Instrumental"


class SeparateStep(PipelineStep):
    name = StepName.SEPARATE
    description = "Separate vocals from background"

    def validate_inputs(self, state: PipelineState) -> None:
        dl = state.get_step(StepName.DOWNLOAD)
        if "audio" not in dl.outputs:
            raise StepValidationError("No audio file from download step")
        audio_path = self._get_audio_path(state)
        if not audio_path.exists():
            raise StepValidationError(f"Audio file not found: {audio_path}")

    def run(self, state: PipelineState) -> PipelineState:
        vocals_path = self.step_dir / VOCALS_FILENAME
        background_path = self.step_dir / BACKGROUND_FILENAME

        # Idempotency: skip if both outputs already exist
        if vocals_path.exists() and background_path.exists():
            log_info("Separation outputs already exist, skipping")
        else:
            audio_path = self._get_audio_path(state)
            log_info(f"Input: {audio_path.name} ({format_file_size(audio_path.stat().st_size)})")

            self._run_separation(audio_path)

            # Verify outputs were created
            if not vocals_path.exists():
                raise SeparationError("Vocals file was not created")
            if not background_path.exists():
                raise SeparationError("Background file was not created")

        vocals_size = format_file_size(vocals_path.stat().st_size)
        bg_size = format_file_size(background_path.stat().st_size)
        log_info(f"Vocals: {vocals_size}, Background: {bg_size}")

        result = state.get_step(self.name)
        result.outputs = {
            "vocals": VOCALS_FILENAME,
            "background": BACKGROUND_FILENAME,
        }
        return state

    # ── Internal methods (mockable in tests) ────────────────────────────────

    def _get_audio_path(self, state: PipelineState) -> Path:
        """Resolve the path to the downloaded audio file."""
        dl = state.get_step(StepName.DOWNLOAD)
        download_dir = self.settings.step_dir(state.video_id, STEP_DIRS[StepName.DOWNLOAD])
        return download_dir / dl.outputs["audio"]

    def _run_separation(self, audio_path: Path) -> None:
        """Run audio-separator with BS-RoFormer model.

        Uses PyTorch with MPS (Metal Performance Shaders) on Apple Silicon.
        When ``separation_use_autocast`` is enabled, audio-separator wraps
        the entire inference in ``torch.autocast("mps", dtype=torch.float16)``
        which halves compute for matmuls/attention (~1.5-2x speedup on M4 Pro).

        Note: audio-separator Roformer models run via PyTorch MPS, NOT ONNX.
        The batch_size param is ignored for Roformer models by audio-separator.
        torch.compile is not used because inductor does not support MPS.
        """
        from audio_separator.separator import Separator

        model_name = self.settings.separation_model
        cache_dir = self.settings.model_cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()

        separator = Separator(
            output_dir=str(self.step_dir),
            model_file_dir=str(cache_dir),
            output_format="WAV",
            normalization_threshold=0.9,
            use_autocast=self.settings.separation_use_autocast,
            mdxc_params={
                "segment_size": self.settings.separation_segment_size,
                "overlap": self.settings.separation_overlap,
                "batch_size": self.settings.separation_batch_size,
            },
        )

        log_info(f"Loading model: {model_name}")
        separator.load_model(model_filename=model_name)
        load_elapsed = time.perf_counter() - t0
        log_info(f"Model loaded in {load_elapsed:.1f}s")

        autocast_status = "enabled" if self.settings.separation_use_autocast else "disabled"
        log_info(f"Separating vocals from background (FP16 autocast: {autocast_status})...")
        t1 = time.perf_counter()
        output_files: list[str] = separator.separate(str(audio_path))
        infer_elapsed = time.perf_counter() - t1
        log_info(f"Inference completed in {infer_elapsed:.1f}s")

        log_info(f"Separator produced {len(output_files)} files")
        self._rename_outputs(output_files)

        # Free memory immediately — these models are large
        del separator
        gc.collect()
        self._cleanup_gpu_memory()

    def _rename_outputs(self, output_files: list[str]) -> None:
        """Rename audio-separator output files to standard names.

        audio-separator produces files like:
          audio_(Vocals)_model_bs_roformer_ep_317.wav
          audio_(Instrumental)_model_bs_roformer_ep_317.wav

        We rename them to vocals.wav and background.wav.
        """
        vocals_path = self.step_dir / VOCALS_FILENAME
        background_path = self.step_dir / BACKGROUND_FILENAME

        vocals_found = False
        instrumental_found = False

        for filepath_str in output_files:
            filepath = Path(filepath_str)
            # audio-separator may return relative paths — resolve against step_dir
            if not filepath.is_absolute():
                filepath = self.step_dir / filepath
            if not filepath.exists():
                log_warning(f"Expected output file not found: {filepath}")
                continue

            name_upper = filepath.name.upper()
            if _VOCALS_LABEL.upper() in name_upper:
                filepath.rename(vocals_path)
                vocals_found = True
            elif _INSTRUMENTAL_LABEL.upper() in name_upper:
                filepath.rename(background_path)
                instrumental_found = True

        if not vocals_found:
            raise SeparationError(f"No vocals stem found in output files: {output_files}")
        if not instrumental_found:
            raise SeparationError(f"No instrumental stem found in output files: {output_files}")

    @staticmethod
    def _cleanup_gpu_memory() -> None:
        """Clear GPU/MPS caches if available."""
        from yt_dbl.utils.memory import cleanup_gpu_memory

        cleanup_gpu_memory()
