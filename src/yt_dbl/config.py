"""Application configuration via pydantic-settings."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["Settings"]


# RAM thresholds for auto-detecting max_loaded_models
_RAM_TIER_VERY_HIGH_GB = 48  # 48+ GB -> batch_size 8
_RAM_TIER_HIGH_GB = 32  # 32+ GB -> 3 models / batch_size 4
_RAM_TIER_MID_GB = 17  # 17-31 GB -> 2 models / batch_size 2


def _get_total_ram_gb() -> float:
    """Return total system RAM in GiB (0.0 on failure)."""
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        return total / (1024**3)
    except (ValueError, OSError, AttributeError):
        return _macos_ram_gb()


def _detect_max_models() -> int:
    """Auto-detect max_loaded_models based on system RAM.

    Heuristic:
      - 16 GB or less: 1 (sequential model loading)
      - 17-31 GB:      2
      - 32 GB+:        3
    """
    total_gb = _get_total_ram_gb()
    if total_gb == 0:
        return 1
    if total_gb >= _RAM_TIER_HIGH_GB:
        return 3
    if total_gb >= _RAM_TIER_MID_GB:
        return 2
    return 1


def _detect_separation_batch_size() -> int:
    """Auto-detect separation batch size based on system RAM.

    Larger batches utilise GPU better during source separation but need
    more memory.  Heuristic:
      - 16 GB or less: 1
      - 17-31 GB:      2
      - 32-47 GB:      4
      - 48 GB+:        8
    """
    total_gb = _get_total_ram_gb()
    if total_gb >= _RAM_TIER_VERY_HIGH_GB:
        return 8
    if total_gb >= _RAM_TIER_HIGH_GB:
        return 4
    if total_gb >= _RAM_TIER_MID_GB:
        return 2
    return 1


def _macos_ram_gb() -> float:
    """Read total RAM on macOS via sysctl, return 0 on failure."""
    import subprocess  # noqa: PLC0415

    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            check=True,
        )
        return int(result.stdout.strip()) / (1024**3)
    except Exception:
        return 0.0


class Settings(BaseSettings):
    """All configurable parameters for yt-dbl.

    Values are loaded from (highest priority first):
      1. CLI arguments / explicit overrides
      2. Environment variables (prefixed YT_DBL_)
      3. .env file
      4. Defaults below
    """

    model_config = SettingsConfigDict(
        env_prefix="YT_DBL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── API keys ────────────────────────────────────────────────────────────
    anthropic_api_key: str = ""
    hf_token: str = ""

    # ── Pipeline ────────────────────────────────────────────────────────────
    target_language: str = "ru"
    output_format: str = "mp4"  # mp4 | mkv
    subtitle_mode: str = "softsub"  # softsub | hardsub | none

    # ── Audio ───────────────────────────────────────────────────────────────
    background_volume: float = Field(default=0.15, ge=0.0, le=1.0)
    background_ducking: bool = True  # reduce background when speech is active
    max_speed_factor: float = Field(default=1.4, ge=1.0, le=2.0)
    voice_ref_duration: float = Field(default=7.0, ge=3.0, le=30.0)
    sample_rate: int = 48000

    # ── Separation ──────────────────────────────────────────────────────────
    # NB: default duplicates registry.SEPARATOR_MODEL (circular import)
    separation_model: str = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
    separation_segment_size: int = Field(default=256, ge=64, le=512)
    separation_overlap: int = Field(default=8, ge=2, le=50)
    separation_batch_size: int = Field(default=0, ge=0, le=16)  # 0 = auto-detect

    # ── Transcription ───────────────────────────────────────────────────────
    transcription_asr_model: str = "mlx-community/VibeVoice-ASR-bf16"
    transcription_aligner_model: str = "mlx-community/Qwen3-ForcedAligner-0.6B-8bit"
    transcription_max_tokens: int = Field(default=8192, ge=256, le=32768)
    transcription_temperature: float = Field(default=0.0, ge=0.0, le=1.0)

    # ── Models ──────────────────────────────────────────────────────────────
    max_loaded_models: int = Field(default=0, ge=0)  # 0 = auto-detect

    @model_validator(mode="after")
    def _auto_detect_from_ram(self) -> Settings:
        """Replace 0 (auto) fields with values based on system RAM."""
        if self.max_loaded_models == 0:
            object.__setattr__(self, "max_loaded_models", _detect_max_models())
        if self.separation_batch_size == 0:
            object.__setattr__(self, "separation_batch_size", _detect_separation_batch_size())
        return self

    # ── Paths ───────────────────────────────────────────────────────────────
    work_dir: Path = Path("work")
    model_cache_dir: Path = Field(
        default_factory=lambda: Path.home() / ".cache" / "yt-dbl" / "models",
    )
    ffmpeg_path: str = ""  # auto-detect: prefers ffmpeg-full if available

    # ── Synthesis (TTS) ──────────────────────────────────────────────────────
    tts_model: str = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"
    tts_temperature: float = Field(default=0.9, ge=0.0, le=2.0)
    tts_top_k: int = Field(default=50, ge=1)
    tts_top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    tts_repetition_penalty: float = Field(default=1.05, ge=1.0, le=2.0)
    tts_sample_rate: int = 24000  # Qwen3-TTS native output rate (model.sample_rate)

    # ── Translation ─────────────────────────────────────────────────────────
    claude_model: str = "claude-sonnet-4-5"

    def job_dir(self, video_id: str) -> Path:
        """Return the working directory for a specific video job."""
        d = self.work_dir / video_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def step_dir(self, video_id: str, step_dirname: str) -> Path:
        """Return the directory for a specific pipeline step."""
        d = self.job_dir(video_id) / step_dirname
        d.mkdir(parents=True, exist_ok=True)
        return d
