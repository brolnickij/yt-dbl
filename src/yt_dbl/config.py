"""Application configuration via pydantic-settings."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    subtitles: bool = True

    # ── Audio ───────────────────────────────────────────────────────────────
    background_volume: float = Field(default=0.15, ge=0.0, le=1.0)
    max_speed_factor: float = Field(default=1.4, ge=1.0, le=2.0)
    voice_ref_duration: float = Field(default=7.0, ge=3.0, le=30.0)
    sample_rate: int = 48000

    # ── Models ──────────────────────────────────────────────────────────────
    max_loaded_models: int = Field(default=1, ge=1)

    # ── Paths ───────────────────────────────────────────────────────────────
    work_dir: Path = Path("work")

    # ── Translation ─────────────────────────────────────────────────────────
    translation_batch_size: int = Field(default=20, ge=1)
    claude_model: str = "claude-sonnet-4-20250514"

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


# Singleton — importable from anywhere
settings = Settings()
