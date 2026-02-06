"""Registry of ML models used by yt-dbl.

Provides model metadata, download status checks, and pre-download capability
by scanning the HuggingFace Hub cache directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "MODEL_REGISTRY",
    "ModelInfo",
    "check_model_downloaded",
    "download_model",
    "get_model_size",
]


@dataclass(frozen=True)
class ModelInfo:
    """Metadata about a model used in the pipeline."""

    repo_id: str  # HuggingFace repo, e.g. "mlx-community/VibeVoice-ASR-bf16"
    purpose: str  # e.g. "ASR + diarization"
    step: str  # Pipeline step that uses it
    approx_size: str  # Human-readable approximate size


# All models the pipeline uses
MODEL_REGISTRY: list[ModelInfo] = [
    ModelInfo(
        repo_id="mlx-community/VibeVoice-ASR-bf16",
        purpose="ASR + speaker diarization",
        step="transcribe",
        approx_size="~8.2 GB",
    ),
    ModelInfo(
        repo_id="mlx-community/Qwen3-ForcedAligner-0.6B-8bit",
        purpose="Word-level forced alignment",
        step="transcribe",
        approx_size="~600 MB",
    ),
    ModelInfo(
        repo_id="mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
        purpose="Text-to-speech with voice cloning",
        step="synthesize",
        approx_size="~1.7 GB",
    ),
]

# audio-separator model (downloaded via its own mechanism, not HF Hub)
_SEPARATOR_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"


def _hf_cache_dir() -> Path:
    """Return the HuggingFace Hub cache directory."""
    import os  # noqa: PLC0415

    # HF_HOME > HF_HUB_CACHE > default
    hf_home = os.environ.get("HF_HOME", "")
    if hf_home:
        return Path(hf_home) / "hub"
    hf_hub = os.environ.get("HF_HUB_CACHE", "")
    if hf_hub:
        return Path(hf_hub)
    return Path.home() / ".cache" / "huggingface" / "hub"


def _repo_dir_name(repo_id: str) -> str:
    """Convert 'mlx-community/VibeVoice-ASR-bf16' → 'models--mlx-community--VibeVoice-ASR-bf16'."""
    return "models--" + repo_id.replace("/", "--")


def check_model_downloaded(repo_id: str) -> bool:
    """Check if a HuggingFace model is downloaded in the cache."""
    cache = _hf_cache_dir()
    repo_dir = cache / _repo_dir_name(repo_id)
    if not repo_dir.exists():
        return False
    # Check for snapshots directory with at least one revision
    snapshots = repo_dir / "snapshots"
    return snapshots.exists() and any(snapshots.iterdir())


def get_model_size(repo_id: str) -> int:
    """Return total size in bytes of a downloaded model, or 0 if not found."""
    cache = _hf_cache_dir()
    repo_dir = cache / _repo_dir_name(repo_id)
    if not repo_dir.exists():
        return 0

    total = 0
    blobs = repo_dir / "blobs"
    if blobs.exists():
        for f in blobs.iterdir():
            if f.is_file():
                total += f.stat().st_size
    return total


def check_separator_downloaded(cache_dir: Path | None = None) -> bool:
    """Check if the audio-separator model is downloaded."""
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "yt-dbl" / "models"
    return (cache_dir / _SEPARATOR_MODEL).exists()


def download_model(repo_id: str) -> None:
    """Download a model from HuggingFace Hub."""
    from huggingface_hub import snapshot_download  # noqa: PLC0415  # type: ignore[import-not-found]

    snapshot_download(repo_id)


def _format_size(size_bytes: int) -> str:
    """Format byte count as human-readable string."""
    if size_bytes == 0:
        return "—"
    gb = size_bytes / (1024**3)
    if gb >= 1.0:
        return f"{gb:.1f} GB"
    mb = size_bytes / (1024**2)
    return f"{mb:.0f} MB"
