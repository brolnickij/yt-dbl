# yt-dbl development tasks
# Run `just` to see all available recipes

set dotenv-load

# Default: show available recipes
default:
    @just --list

# ── Quality ─────────────────────────────────────────────────────────────────

# Run all checks (lint + format + typecheck + tests)
check: lint format-check typecheck test

# Ruff lint
lint:
    uv run ruff check src/ tests/

# Ruff lint with auto-fix
fix:
    uv run ruff check --fix src/ tests/

# Check formatting
format-check:
    uv run ruff format --check src/ tests/

# Auto-format code
format:
    uv run ruff format src/ tests/

# Mypy type checking
typecheck:
    uv run mypy src/ tests/

# ── Testing ─────────────────────────────────────────────────────────────────

# Run fast tests (parallel, with coverage)
test:
    uv run pytest tests/ --cov -q

# Run tests verbose
test-v:
    uv run pytest tests/ --cov --cov-report=term-missing

# Run E2E tests (requires network + ffmpeg)
test-e2e:
    uv run pytest tests/e2e/ --run-slow --timeout=120 -v

# Run all tests including E2E
test-all: test test-e2e

# Run tests matching a pattern
test-k pattern:
    uv run pytest tests/ -k "{{ pattern }}" -v

# ── Dev ─────────────────────────────────────────────────────────────────────

# Install/sync dependencies
sync:
    uv sync

# Run pre-commit on all files
pre-commit:
    uv run pre-commit run --all-files

# Install pre-commit hooks
hooks:
    uv run pre-commit install

# ── Run ─────────────────────────────────────────────────────────────────────

# Run the CLI
run *args:
    uv run yt-dbl {{ args }}

# Dub a video
dub url lang="ru":
    uv run yt-dbl dub "{{ url }}" --lang {{ lang }}

# Resume a pipeline
resume video_id:
    uv run yt-dbl resume {{ video_id }}

# Check pipeline status
status video_id:
    uv run yt-dbl status {{ video_id }}
