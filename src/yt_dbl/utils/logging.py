"""Rich-based logging and progress utilities."""

from __future__ import annotations

import logging
import os
import time
import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.theme import Theme

from yt_dbl.schemas import STEP_ORDER, StepName

if TYPE_CHECKING:
    from collections.abc import Generator


# ── Memory measurement ──────────────────────────────────────────────────────


def get_rss_mb() -> float:
    """Return peak process RSS in megabytes (ru_maxrss)."""
    try:
        import resource  # noqa: PLC0415

        # macOS/Linux: ru_maxrss is in bytes on Linux, kilobytes on macOS
        usage = resource.getrusage(resource.RUSAGE_SELF)
        import sys  # noqa: PLC0415

        if sys.platform == "darwin":
            return usage.ru_maxrss / (1024 * 1024)  # bytes → MB
        return usage.ru_maxrss / 1024  # KB → MB
    except ImportError:
        return 0.0


def get_metal_memory_mb() -> float:
    """Return MLX Metal active memory in megabytes (0 if unavailable)."""
    try:
        import mlx.core as mx  # noqa: PLC0415

        if hasattr(mx, "metal") and hasattr(mx.metal, "get_active_memory"):
            return float(mx.metal.get_active_memory()) / (1024 * 1024)
    except ImportError:
        pass
    return 0.0


def log_memory_status() -> None:
    """Print current memory usage."""
    rss = get_rss_mb()
    metal = get_metal_memory_mb()
    parts = [f"  [info]Memory: RSS={rss:.0f} MB"]
    if metal > 0:
        parts.append(f"Metal={metal:.0f} MB")
    console.print(", ".join(parts) + "[/info]")


# ── Console ─────────────────────────────────────────────────────────────────

theme = Theme(
    {
        "step": "bold cyan",
        "success": "bold green",
        "warning": "bold yellow",
        "error": "bold red",
        "info": "dim",
        "model": "bold magenta",
    }
)

console = Console(theme=theme)

# ── Step numbering ──────────────────────────────────────────────────────────

TOTAL_STEPS = len(STEP_ORDER)


def step_prefix(step: StepName) -> str:
    """Return formatted step prefix like '[1/6]'."""
    idx = STEP_ORDER.index(step) + 1
    return f"[step]\\[{idx}/{TOTAL_STEPS}][/step]"


def step_label(step: StepName) -> str:
    """Human-readable step name."""
    labels = {
        StepName.DOWNLOAD: "Download",
        StepName.SEPARATE: "Separate",
        StepName.TRANSCRIBE: "Transcribe",
        StepName.TRANSLATE: "Translate",
        StepName.SYNTHESIZE: "Synthesize",
        StepName.ASSEMBLE: "Assemble",
    }
    return labels[step]


# ── Logging helpers ─────────────────────────────────────────────────────────


def log_step_start(step: StepName, detail: str = "") -> None:
    msg = f"{step_prefix(step)} {step_label(step)}"
    if detail:
        msg += f" — {detail}"
    console.print(msg)


def log_step_done(step: StepName, elapsed: float) -> None:
    console.print(
        f"{step_prefix(step)} {step_label(step)} [success]✓[/success] [info]({elapsed:.1f}s)[/info]"
    )


def log_step_skip(step: StepName) -> None:
    console.print(f"{step_prefix(step)} {step_label(step)} [info]— skipped (checkpoint)[/info]")


def log_step_fail(step: StepName, error: str) -> None:
    console.print(f"{step_prefix(step)} {step_label(step)} [error]✗ {error}[/error]")


def log_info(msg: str) -> None:
    console.print(f"  [info]{msg}[/info]")


def log_warning(msg: str) -> None:
    console.print(f"  [warning]⚠ {msg}[/warning]")


def format_file_size(size_bytes: int) -> str:
    """Format byte count as human-readable string (e.g. ``142.3 MB``)."""
    if size_bytes == 0:
        return "0 B"
    gb = size_bytes / (1024**3)
    if gb >= 1.0:
        return f"{gb:.1f} GB"
    mb = size_bytes / (1024**2)
    return f"{mb:.1f} MB"


def log_model_load(name: str, elapsed: float = 0.0, mem_delta_mb: float = 0.0) -> None:
    parts = [f"  [model]↑ Loading model:[/model] {name}"]
    if elapsed > 0:
        parts.append(f"[info]({elapsed:.1f}s)[/info]")
    if mem_delta_mb > 0:
        parts.append(f"[info]+{mem_delta_mb:.0f} MB[/info]")
    console.print(" ".join(parts))


def log_model_unload(name: str, mem_freed_mb: float = 0.0) -> None:
    parts = [f"  [model]↓ Unloading model:[/model] {name}"]
    if mem_freed_mb > 0:
        parts.append(f"[info]-{mem_freed_mb:.0f} MB[/info]")
    console.print(" ".join(parts))


# ── Progress bars ───────────────────────────────────────────────────────────


def create_progress() -> Progress:
    """Create a standard progress bar for pipeline operations."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    )


# ── Library noise suppression ───────────────────────────────────────────────

_NOISY_LOGGERS = (
    "huggingface_hub",
    "transformers",
    "tokenizers",
    "mlx_audio",
    "tqdm",
    "filelock",
)


@contextmanager
def suppress_library_noise() -> Generator[None, None, None]:
    """Silence noisy HuggingFace / transformers / tqdm output during model loading.

    Suppresses:
    - HF Hub download progress bars
    - "You are using a model of type ..." warnings
    - Tokenizer regex warnings
    - tqdm progress bars
    """
    old_env: dict[str, str | None] = {}
    env_overrides = {
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    for key, val in env_overrides.items():
        old_env[key] = os.environ.get(key)
        os.environ[key] = val

    saved_levels: dict[str, int] = {}
    for name in _NOISY_LOGGERS:
        logger = logging.getLogger(name)
        saved_levels[name] = logger.level
        logger.setLevel(logging.ERROR)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*model of type.*")
        warnings.filterwarnings("ignore", message=".*incorrect regex pattern.*")
        warnings.filterwarnings("ignore", message=".*fix_mistral_regex.*")
        try:
            yield
        finally:
            for name, level in saved_levels.items():
                logging.getLogger(name).setLevel(level)
            for key, old_val in old_env.items():
                if old_val is not None:
                    os.environ[key] = old_val
                else:
                    os.environ.pop(key, None)


# ── Timer context manager ──────────────────────────────────────────────────


@contextmanager
def timer() -> Generator[dict[str, float], None, None]:
    """Context manager that tracks elapsed time.

    Usage:
        with timer() as t:
            do_something()
        print(t["elapsed"])
    """
    result: dict[str, float] = {"elapsed": 0.0}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["elapsed"] = time.perf_counter() - start
