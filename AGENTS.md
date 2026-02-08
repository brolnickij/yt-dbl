# AGENTS.md

> Instructions for AI coding agents working on **yt-dbl** — a CLI tool for automatic YouTube video dubbing with voice cloning on Apple Silicon.

---

## Project overview

`yt-dbl` downloads a YouTube video, separates vocals from background, transcribes and diarizes speakers, translates via Claude API, synthesizes dubbed speech with voice cloning (Qwen3-TTS), and assembles the final video — all locally on Apple Silicon via MLX.

- **Language:** Python 3.12+, fully typed (`py.typed`, strict mypy)
- **Platform:** macOS Apple Silicon only (M1–M4); ML inference runs on Metal GPU via MLX
- **Package manager:** [uv](https://docs.astral.sh/uv/)
- **Build backend:** Hatchling
- **CLI framework:** Typer + Rich
- **Config:** pydantic-settings (`Settings` class in `src/yt_dbl/config.py`)
- **Data models:** Pydantic v2 (`src/yt_dbl/schemas.py`)
- **Task runner:** [just](https://github.com/casey/just) (see `justfile`)

---

## Repository structure

```
src/yt_dbl/              # Main package (src-layout)
├── __init__.py          # __version__ only
├── cli.py               # Typer CLI: dub, resume, status, models list/download
├── config.py            # Settings (pydantic-settings, env prefix YT_DBL_)
├── schemas.py           # Pydantic models: PipelineState, Segment, Speaker, Word, StepName
├── models/              # ML model lifecycle
│   ├── manager.py       # LRU ModelManager (load/unload/evict)
│   └── registry.py      # HuggingFace model registry + download helpers
├── pipeline/            # 6-step pipeline
│   ├── base.py          # PipelineStep ABC + exception hierarchy
│   ├── runner.py        # PipelineRunner: orchestration, checkpointing, resume
│   ├── download.py      # Step 1: yt-dlp download + ffmpeg audio extraction
│   ├── separate.py      # Step 2: BS-RoFormer vocal/background separation (PyTorch MPS)
│   ├── transcribe.py    # Step 3: VibeVoice-ASR + Qwen3-ForcedAligner (MLX)
│   ├── translate.py     # Step 4: Claude API translation (auto-batched)
│   ├── synthesize.py    # Step 5: Qwen3-TTS voice cloning + postprocessing (MLX)
│   └── assemble.py      # Step 6: speech track + background + video muxing (ffmpeg)
└── utils/
    ├── audio.py             # ffmpeg wrapper (detect, run, extract, duration)
    ├── audio_processing.py  # Voice ref extraction, speed-up, loudnorm, de-essing
    ├── languages.py         # Language code maps (ALIGNER_LANGUAGE_MAP, TTS_LANG_MAP)
    ├── logging.py           # Rich console, step logging, progress bars, memory tracking
    └── memory.py            # GPU memory cleanup (MLX Metal, PyTorch MPS/CUDA)

tests/                   # Test pyramid
├── conftest.py          # Root fixtures: env isolation, prefill_* helpers, E2E cache
├── unit/                # Fast, isolated, no I/O
├── integration/         # Cross-module (runner, CLI), mocked external deps
└── e2e/                 # Real ML models + network, gated by --run-slow
```

---

## Setup & environment

```bash
# Clone and install
git clone git@github.com:brolnickij/yt-dbl.git && cd yt-dbl
uv sync

# System dependencies (required for E2E/real use)
brew install ffmpeg yt-dlp
```

The project uses a **virtual environment** managed by `uv` (`.venv/`). Always run commands through `uv run` or activate the venv first.

---

## Build, lint, test commands

All tasks are defined in the `justfile`. Prefer `just` commands over raw invocations.

| Task | Command | What it does |
|---|---|---|
| **All checks** | `just check` | lint + format-check + typecheck + test |
| **Lint** | `just lint` | `ruff check src/ tests/` |
| **Auto-fix** | `just fix` | `ruff check --fix src/ tests/` |
| **Format check** | `just format-check` | `ruff format --check src/ tests/` |
| **Auto-format** | `just format` | `ruff format src/ tests/` |
| **Type check** | `just typecheck` | `mypy src/ tests/` (strict mode) |
| **Fast tests** | `just test` | `pytest tests/ --cov -q` (parallel, with coverage) |
| **Verbose tests** | `just test-v` | Same + `--cov-report=term-missing` |
| **E2E tests** | `just test-e2e` | `pytest tests/e2e/ --run-slow --timeout=300 -n0 -v` |
| **All tests** | `just test-all` | unit + integration + E2E |
| **Pattern match** | `just test-k <pattern>` | `pytest tests/ -k "<pattern>" -v` |
| **Run CLI** | `just run <args>` | `uv run yt-dbl <args>` |

### Verification sequence

After any code change, always run the full quality gate:

```bash
just check
```

This runs: `ruff check` → `ruff format --check` → `mypy --strict` → `pytest` (parallel with coverage).

CI runs the same checks on every push/PR to `master`.

---

## Code style & conventions

### General rules

- **Line length:** 100 characters
- **Target Python version:** 3.11 (ruff) / 3.12+ (runtime)
- **Imports:** isort-ordered, `from __future__ import annotations` in every module
- **String quotes:** double quotes (enforced by ruff format)
- **Type annotations:** required on all public functions and methods (mypy strict)
- **Docstrings:** module-level docstrings on every file, class/method docstrings on public API
- **`__all__`:** exported in all public modules
- **Constants:** `UPPER_SNAKE_CASE`, private helpers prefixed with `_`

### Naming conventions

- Pipeline steps: `<Verb>Step` (e.g., `DownloadStep`, `TranscribeStep`)
- Errors: `<Noun>Error` inheriting from `PipelineStepError`
- Test classes: `Test<Component>` (e.g., `TestSettings`, `TestPipelineRunner`)
- Test functions: `test_<behavior_being_tested>` (e.g., `test_resume_skips_completed`)
- Fixtures: snake_case descriptive nouns (e.g., `work_dir`, `pipeline_state`, `make_settings`)
- Prefill helpers: `prefill_<step_name>()` — utility functions (not fixtures) to populate pipeline state

### Lazy imports

Heavy dependencies (torch, mlx, anthropic, audio_separator, soundfile, scipy, numpy) are imported **lazily inside functions**, not at module level. This keeps CLI startup fast (<100 ms). The `PLC0415` ruff rule is selectively disabled per file in `ruff.toml` for modules that use lazy imports.

### Error handling

- Pipeline steps use a typed exception hierarchy rooted at `PipelineStepError`
- Each step has its own error class: `DownloadError`, `SeparationError`, `TranscriptionError`, `TranslationError`, `SynthesisError`, `AssemblyError`
- `StepValidationError` for pre-flight input checks in `validate_inputs()`
- The runner catches all exceptions, marks the step as `FAILED`, persists state, and stops

### Idempotency

Every pipeline step checks for existing output files before doing work. Re-running a step that already produced valid output is a no-op. This is critical for the `resume` command and for `--from-step` reruns.

---

## Architecture patterns

### Pipeline step contract

Every step extends `PipelineStep` (ABC) and implements:

1. `name: StepName` — enum identifier
2. `description: str` — human-readable label for logs
3. `validate_inputs(state)` — pre-flight checks, raises `StepValidationError`
4. `run(state) -> PipelineState` — main logic, must be idempotent

Steps receive `Settings`, a `step_dir` (Path), and an optional `ModelManager`.

### State & checkpointing

- `PipelineState` is the single source of truth, serialized to `state.json` (Pydantic → JSON)
- State is saved atomically (temp file + `os.replace`) after every step (success or failure)
- `load_state()` / `save_state()` in `pipeline/runner.py`
- Steps record their outputs as relative paths in `StepResult.outputs`

### Model lifecycle

- `ModelManager` is an LRU cache: loads models on demand, evicts least-recently-used when capacity is exceeded
- Models are registered with `loader` and `unloader` callables
- Capacity (`max_loaded_models`) is auto-detected from system RAM (1–3 models)
- After each unload: `gc.collect()` + MLX/PyTorch cache clearing via `utils/memory.py`

### Configuration priority

1. CLI arguments (explicit overrides)
2. Environment variables (prefix `YT_DBL_`)
3. `.env` file (pydantic-settings)
4. Default values in `Settings`

---

## Testing guidelines

### Test pyramid

- **Unit tests** (`tests/unit/`): fast, isolated, no I/O, no network. Mock all external dependencies.
- **Integration tests** (`tests/integration/`): test cross-module interactions (runner, CLI). Mock ML models and external tools.
- **E2E tests** (`tests/e2e/`): real ML inference + network. Gated by `@pytest.mark.slow` and `--run-slow` flag.

### Key testing patterns

- **Environment isolation:** the root `_no_dotenv` autouse fixture changes CWD to a clean tmpdir and strips all `YT_DBL_*` env vars so `Settings` is fully isolated in every test
- **Prefill helpers:** `prefill_download()`, `prefill_separate()`, `prefill_transcribe()`, `prefill_translate()`, `prefill_synthesize()` — utility functions in root `conftest.py` that populate a `PipelineState` and create fake files for a given step. Call them in sequence to build up state for later steps
- **Factory fixtures:** `make_settings(...)` returns a `Settings` bound to `tmp_path`
- **Parametrize:** use `@pytest.mark.parametrize` for data-driven tests (RAM tiers, language detection, format sizes)
- **E2E sequential:** all E2E tests share a single xdist group (`e2e_sequential`) via conftest hook to prevent parallel GPU model loading

### Pytest configuration

- `pytest.ini`: `--strict-markers`, `--strict-config`, `--timeout=30`, `-x` (fail-fast), `-n auto` (parallel via xdist), `--dist loadgroup`, `-p randomly`
- Marker: `@pytest.mark.slow` — skipped unless `--run-slow` is passed
- `pythonpath = src` — allows direct imports like `from yt_dbl.config import Settings`

### Writing new tests

1. Place in the correct layer: `unit/` for isolated logic, `integration/` for cross-module, `e2e/` for real inference
2. Always use `tmp_path` or `work_dir` fixture for file operations — never write to the real filesystem
3. Mock heavy dependencies: `audio_separator`, `mlx_audio`, `anthropic`, `subprocess` calls
4. For pipeline step tests, use `prefill_*` helpers to set up prerequisite state
5. Run `just check` before committing

---

## Commit conventions

This project uses **Conventional Commits** (enforced by pre-commit hook):

```
<type>(<scope>): <description>

Types: feat, fix, refactor, style, build, ci, docs, test, perf, chore, revert
Scope: optional, usually a module name (e.g., translate, config, cli, synthesize)
```

Examples:
- `feat(translate): add auto-batching for long audio translation`
- `fix(synthesize): correct TTS sample rate from 12kHz to 24kHz`
- `test: add coverage for utils/memory.py GPU cleanup`
- `perf(transcribe): load audio once and slice per segment for alignment`

Releases are automated via **python-semantic-release** on the `master` branch. Version is tracked in `pyproject.toml` and `src/yt_dbl/__init__.py`.

---

## CI/CD

- **CI** (`.github/workflows/ci.yml`): runs on every push/PR to `master`
  - Lint (ruff check), format check (ruff format), type check (mypy strict)
  - Unit + integration tests with coverage (xdist parallel)
  - E2E tests are NOT run in CI (require Apple Silicon + GPU)
  - Uses `UV_TORCH_BACKEND=cpu` to avoid downloading CUDA/Metal torch in CI
- **Release** (`.github/workflows/release.yml`): triggered on push to `master`, runs CI first, then python-semantic-release + PyPI publish

---

## Key domain knowledge

### Pipeline steps (in order)

| # | Step | Model / Tool | Output |
|---|---|---|---|
| 1 | Download | yt-dlp + ffmpeg | `video.mp4`, `audio.wav` (48 kHz mono) |
| 2 | Separate | BS-RoFormer (PyTorch MPS) | `vocals.wav`, `background.wav` |
| 3 | Transcribe | VibeVoice-ASR 4-bit (~5.7 GB) + Qwen3-ForcedAligner (~600 MB) | `segments.json` |
| 4 | Translate | Claude Sonnet 4.5 (Anthropic API) | `translations.json`, `subtitles.srt` |
| 5 | Synthesize | Qwen3-TTS (~1.7 GB, MLX) | `segment_NNNN.wav`, `synth_meta.json` |
| 6 | Assemble | ffmpeg (speech track + sidechain ducking + mux) | `result.mp4` |

### Audio chain

- TTS native rate: **24 kHz** (Qwen3-TTS) → postprocessing resamples to **48 kHz**
- Postprocessing per segment: optional speed-up (rubberband or atempo) → 2-pass loudnorm (-16 LUFS) → de-essing
- Final assembly: equal-power crossfade (50 ms) + sidechain ducking of background + AAC 320 kbps

### Memory management

- ML models are large (5.7 GB ASR + 1.7 GB TTS + 600 MB aligner)
- LRU ModelManager evicts least-recently-used model when `max_loaded_models` is reached
- ASR is explicitly unloaded before loading the aligner to avoid holding both in memory
- After separation: `gc.collect()` + `torch.mps.empty_cache()`
- After all pipeline steps: `model_manager.unload_all()`

### Language detection

- Unicode-script heuristics (Cyrillic → `ru`, kana → `ja`, hangul → `ko`, CJK → `zh`, etc.)
- Japanese kana is checked before CJK ideographs (kanji overlap with Chinese)
- Falls back to `en` for Latin scripts

---

## Common pitfalls

1. **Never import heavy ML libs at module level** — use lazy imports inside functions. Fast CLI startup is a hard requirement.
2. **Always test idempotency** — steps must produce the same result and skip work if outputs already exist.
3. **Respect the test matrix** — ruff ignores (like `PLC0415`, `S603`) are set per-file in `ruff.toml`. Don't suppress globally what's suppressed selectively.
4. **Atomic state writes** — `save_state()` uses temp file + `os.replace`. Never write `state.json` directly.
5. **E2E tests require Apple Silicon** — they load real GPU models. They are gated by `--run-slow` and excluded from CI.
6. **Separation batch_size** — despite being configurable, audio-separator ignores it for Roformer models. It's kept for future compatibility.
7. **Pre-release dependencies** — `transformers>=5.0.0rc3` is a pre-release; `uv` needs `prerelease = "allow"` in `[tool.uv]`.
8. **Cyrillic Unicode** — `RUF001`/`RUF003` are deliberately ignored. Russian text with stress marks (acute accent U+0301) is intentional for TTS quality.
9. **ffmpeg-full vs ffmpeg** — the tool auto-detects brew's keg-only `ffmpeg-full` for rubberband support, falling back to system ffmpeg. Detection is cached via `@lru_cache`.
10. **State is the contract** — `PipelineState` is the sole interface between steps. Steps must not reach into other steps' directories directly; use `resolve_step_file()`.
