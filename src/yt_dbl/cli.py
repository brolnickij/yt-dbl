"""CLI interface for yt-dbl."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from yt_dbl import __version__
from yt_dbl.config import Settings
from yt_dbl.schemas import STEP_DIRS, STEP_ORDER, PipelineState, StepName, StepStatus
from yt_dbl.utils.logging import console, log_warning

_OutputDir = Annotated[
    Path | None,
    typer.Option("-o", "--output-dir", help="Output directory (default: ./dubbed)"),
]

app = typer.Typer(
    name="yt-dbl",
    help="YouTube video dubbing with voice cloning.",
    no_args_is_help=True,
    rich_markup_mode="rich",
    add_completion=False,
)
models_app = typer.Typer(help="Manage ML models.", no_args_is_help=True)
app.add_typer(models_app, name="models")


# ── Helpers ─────────────────────────────────────────────────────────────────


def _extract_video_id(url: str) -> str:
    """Extract YouTube video ID from various URL formats."""
    patterns = [
        r"(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"(?:shorts/)([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    # Fallback: use sanitized URL as ID
    return re.sub(r"[^\w-]", "_", url)[:64]


def _make_settings(**overrides: object) -> Settings:
    """Create Settings with CLI overrides applied."""
    override_dict = {k: v for k, v in overrides.items() if v is not None}
    base = Settings()
    return base.model_copy(update=override_dict) if override_dict else base


def _step_name_from_str(step_str: str) -> StepName:
    """Convert a step string to StepName enum."""
    try:
        return StepName(step_str.lower())
    except ValueError as err:
        valid = ", ".join(s.value for s in StepName)
        raise typer.BadParameter(f"Unknown step '{step_str}'. Valid: {valid}") from err


def _invalidate_language_steps(
    state: PipelineState,
    cfg: Settings,
    video_id: str,
) -> None:
    """Reset language-dependent steps and remove their cached outputs."""
    for step_name in (StepName.TRANSLATE, StepName.SYNTHESIZE, StepName.ASSEMBLE):
        step_result = state.steps.get(step_name)
        if step_result:
            step_result.status = StepStatus.PENDING
            step_result.outputs.clear()
            step_result.error = ""
            step_result.duration_sec = 0.0
        step_dir = cfg.work_dir / video_id / STEP_DIRS[step_name]
        if step_dir.exists():
            shutil.rmtree(step_dir)
    # Remove result files from job root
    job_dir = cfg.work_dir / video_id
    for ext in ("mp4", "mkv"):
        result_file = job_dir / f"result.{ext}"
        if result_file.exists():
            result_file.unlink()


# ── Commands ────────────────────────────────────────────────────────────────


@app.command()
def dub(
    url: Annotated[str, typer.Argument(help="YouTube video URL")],
    target_language: Annotated[
        str | None, typer.Option("-t", "--target-language", help="Target language")
    ] = None,
    max_models: Annotated[
        int | None, typer.Option("--max-models", help="Max models in memory")
    ] = None,
    from_step: Annotated[
        str | None, typer.Option("--from-step", help="Start from this step")
    ] = None,
    background_volume: Annotated[
        float | None, typer.Option("--bg-volume", help="Background volume (0.0-1.0)")
    ] = None,
    max_speed: Annotated[
        float | None, typer.Option("--max-speed", help="Max TTS speed factor")
    ] = None,
    no_subs: Annotated[bool, typer.Option("--no-subs", help="Disable subtitles")] = False,
    sub_mode: Annotated[
        str | None,
        typer.Option("--sub-mode", help="Subtitle mode (softsub/hardsub/none)"),
    ] = None,
    output_format: Annotated[
        str | None, typer.Option("--format", help="Output format (mp4/mkv)")
    ] = None,
    output_dir: _OutputDir = None,
) -> None:
    """Dub a YouTube video with translated voice cloning."""
    from yt_dbl.pipeline.runner import PipelineRunner, load_state, save_state

    cfg = _make_settings(
        target_language=target_language,
        max_loaded_models=max_models,
        background_volume=background_volume,
        max_speed_factor=max_speed,
        subtitle_mode="none" if no_subs else sub_mode,
        output_format=output_format,
        work_dir=output_dir,
    )

    video_id = _extract_video_id(url)

    # Try to load existing state (for implicit resume)
    state = load_state(cfg, video_id)
    if state is None:
        state = PipelineState(
            video_id=video_id,
            url=url,
            target_language=cfg.target_language,
        )
        save_state(state, cfg)
    elif target_language is not None and target_language != state.target_language:
        log_warning(f"Target language changed: {state.target_language} → {target_language}")
        state.target_language = target_language
        _invalidate_language_steps(state, cfg, video_id)
        save_state(state, cfg)

    from_step_enum = _step_name_from_str(from_step) if from_step else None

    runner = PipelineRunner(cfg)
    runner.run(state, from_step=from_step_enum)


@app.command()
def resume(
    video_id: Annotated[str, typer.Argument(help="Video ID to resume")],
    max_models: Annotated[
        int | None, typer.Option("--max-models", help="Max models in memory")
    ] = None,
    output_dir: _OutputDir = None,
) -> None:
    """Resume a previously interrupted dubbing job."""
    from yt_dbl.pipeline.runner import PipelineRunner, load_state

    cfg = _make_settings(max_loaded_models=max_models, work_dir=output_dir)

    state = load_state(cfg, video_id)
    if state is None:
        console.print(f"[error]No job found for video ID: {video_id}[/error]")
        raise typer.Exit(1)

    next_step = state.next_step
    if next_step is None:
        console.print("[success]Job already completed![/success]")
        raise typer.Exit(0)

    console.print(f"Resuming from step: [step]{next_step.value}[/step]")
    runner = PipelineRunner(cfg)
    runner.run(state)


@app.command()
def status(
    video_id: Annotated[str, typer.Argument(help="Video ID to check")],
    output_dir: _OutputDir = None,
) -> None:
    """Show the status of a dubbing job."""
    from yt_dbl.pipeline.runner import load_state

    cfg = _make_settings(work_dir=output_dir)
    state = load_state(cfg, video_id)
    if state is None:
        console.print(f"[error]No job found for video ID: {video_id}[/error]")
        raise typer.Exit(1)

    table = Table(title=f"Job: {video_id}")
    table.add_column("Step", style="cyan")
    table.add_column("Status")
    table.add_column("Duration", justify="right")
    table.add_column("Error")

    for step_name in STEP_ORDER:
        result = state.steps.get(step_name)
        if result is None:
            table.add_row(step_name.value, "[dim]pending[/dim]", "", "")
            continue

        status_style = {
            StepStatus.PENDING: "[dim]pending[/dim]",
            StepStatus.RUNNING: "[yellow]running[/yellow]",
            StepStatus.COMPLETED: "[green]completed[/green]",
            StepStatus.FAILED: "[red]failed[/red]",
        }
        dur = f"{result.duration_sec:.1f}s" if result.duration_sec else ""
        err = result.error[:50] if result.error else ""
        table.add_row(step_name.value, status_style[result.status], dur, err)

    console.print(table)

    if state.meta:
        console.print(f"\n  Title: {state.meta.title}")
        console.print(f"  Channel: {state.meta.channel}")
        console.print(f"  Duration: {state.meta.duration:.0f}s")


# ── Models subcommands ──────────────────────────────────────────────────────


@models_app.command("list")
def models_list() -> None:
    """Show status of ML models."""
    from yt_dbl.models.registry import (
        MODEL_REGISTRY,
        SEPARATOR_MODEL,
        check_model_downloaded,
        check_separator_downloaded,
        format_model_size,
        get_model_size,
    )

    table = Table(title="ML Models")
    table.add_column("Model", style="magenta")
    table.add_column("Purpose")
    table.add_column("Status")
    table.add_column("Size", justify="right")

    total_size = 0

    for info in MODEL_REGISTRY:
        downloaded = check_model_downloaded(info.repo_id)
        size_bytes = get_model_size(info.repo_id) if downloaded else 0
        total_size += size_bytes

        status_str = "[green]✓ downloaded[/green]" if downloaded else "[dim]not downloaded[/dim]"
        size_str = format_model_size(size_bytes) if downloaded else info.approx_size

        table.add_row(info.repo_id.split("/")[-1], info.purpose, status_str, size_str)

    # Separator model (non-HF)
    cfg = Settings()
    sep_downloaded = check_separator_downloaded(cfg.model_cache_dir)
    sep_status = "[green]✓ downloaded[/green]" if sep_downloaded else "[dim]not downloaded[/dim]"
    sep_size = ""
    if sep_downloaded:
        sep_path = cfg.model_cache_dir / SEPARATOR_MODEL
        sep_bytes = sep_path.stat().st_size
        total_size += sep_bytes
        sep_size = format_model_size(sep_bytes)
    else:
        sep_size = "~200 MB"

    table.add_row("MelBand-RoFormer", "Vocal separation", sep_status, sep_size)

    console.print(table)

    if total_size > 0:
        console.print(f"\n  Total cached: {format_model_size(total_size)}")
    console.print("  [info]Use 'yt-dbl models download' to pre-download all models[/info]")


@models_app.command("download")
def models_download() -> None:
    """Pre-download all required ML models."""
    from yt_dbl.models.registry import MODEL_REGISTRY, check_model_downloaded, download_model

    for info in MODEL_REGISTRY:
        if check_model_downloaded(info.repo_id):
            console.print(f"  [green]✓[/green] {info.repo_id} — already downloaded")
            continue

        console.print(f"  [info]↓ Downloading {info.repo_id} ({info.approx_size})...[/info]")
        try:
            download_model(info.repo_id)
            console.print(f"  [green]✓[/green] {info.repo_id}")
        except Exception as exc:
            console.print(f"  [error]✗ {info.repo_id}: {exc}[/error]")

    console.print("\n  [info]Separator model downloads automatically on first use.[/info]")
    console.print("[success]Done![/success]")


# ── Version ─────────────────────────────────────────────────────────────────


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"yt-dbl {__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option("--version", "-V", callback=_version_callback, is_eager=True),
    ] = None,
) -> None:
    """YouTube video dubbing with voice cloning."""
