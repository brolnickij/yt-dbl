"""Rich-based logging and progress utilities."""

from __future__ import annotations

import time
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


def log_model_load(name: str) -> None:
    console.print(f"  [model]↑ Loading model:[/model] {name}")


def log_model_unload(name: str) -> None:
    console.print(f"  [model]↓ Unloading model:[/model] {name}")


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
