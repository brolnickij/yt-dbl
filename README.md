# yt-dbl
> [!WARNING]
> **Apple Silicon only** (M1/M2/M3/M4) — all ML inference runs on Metal GPU via MLX
>
> Tested on **M4 Pro** (20-core GPU, 48 GB unified memory)

CLI tool for automatic YouTube video dubbing with voice cloning.

All ML inference (ASR, alignment, TTS) runs locally on Apple Silicon via [MLX](https://github.com/ml-explore/mlx). Translation is done through the Claude API. The output is a video file dubbed in the target language using the original speaker's cloned voice.

## Requirements
- **macOS** with **Apple Silicon** (M1/M2/M3/M4) — MLX only works on Metal
- **Python** >= 3.11
- **FFmpeg** — auto-detected; prefers `ffmpeg-full` (Homebrew) for rubberband support
- **yt-dlp** — video downloading
- **Anthropic API key** — for translation via Claude

## Installation
```bash
git clone git@github.com:lucidrains/BS-RoFormer.git && cd yt-dbl
uv sync

# API key (required for the translation step)
export YT_DBL_ANTHROPIC_API_KEY="sk-ant-..."

# Optional: ffmpeg-full for pitch-preserving speed-up (rubberband)
brew install ffmpeg-full
```

### Pre-downloading models
Models are downloaded automatically on first run, but you can fetch them ahead of time (~10.5 GB):

```bash
uv run yt-dbl models download
```

## Configuration
Settings are loaded in order of priority:

1. CLI arguments
2. Environment variables (prefix `YT_DBL_`)
3. `.env` file
4. Default values

Example `.env`:

```env
YT_DBL_ANTHROPIC_API_KEY=sk-ant-...
YT_DBL_TARGET_LANGUAGE=ru
YT_DBL_BACKGROUND_VOLUME=0.2
YT_DBL_MAX_SPEED_FACTOR=1.3
YT_DBL_OUTPUT_FORMAT=mkv
YT_DBL_SUBTITLE_MODE=hardsub
YT_DBL_BACKGROUND_DUCKING=true
YT_DBL_VOICE_REF_DURATION=7.0
YT_DBL_SAMPLE_RATE=48000
```

### All parameters
| Parameter | Env variable | Default | Description |
|---|---|---|---|
| `target_language` | `YT_DBL_TARGET_LANGUAGE` | `ru` | Target language |
| `output_format` | `YT_DBL_OUTPUT_FORMAT` | `mp4` | `mp4` / `mkv` |
| `subtitle_mode` | `YT_DBL_SUBTITLE_MODE` | `softsub` | `softsub` / `hardsub` / `none` |
| `background_volume` | `YT_DBL_BACKGROUND_VOLUME` | `0.15` | Background volume (0.0–1.0) |
| `background_ducking` | `YT_DBL_BACKGROUND_DUCKING` | `true` | Duck background during speech (sidechain) |
| `max_speed_factor` | `YT_DBL_MAX_SPEED_FACTOR` | `1.4` | Max TTS speed-up (1.0–2.0) |
| `voice_ref_duration` | `YT_DBL_VOICE_REF_DURATION` | `7.0` | Voice reference duration (3–30 sec) |
| `max_loaded_models` | `YT_DBL_MAX_LOADED_MODELS` | `0` (auto) | Max models in memory |
| `anthropic_api_key` | `YT_DBL_ANTHROPIC_API_KEY` | — | Anthropic API key |
| `work_dir` | `YT_DBL_WORK_DIR` | `work` | Working directory |

## Quick start
```bash
# Dub a video into Russian (default)
uv run yt-dbl dub "https://www.youtube.com/watch?v=VIDEO_ID"

# Specify target language
uv run yt-dbl dub "https://youtu.be/VIDEO_ID" -t es

# Start from a specific step (previous steps are skipped)
uv run yt-dbl dub "https://youtu.be/VIDEO_ID" --from-step translate

# Check job status
uv run yt-dbl status VIDEO_ID

# Resume an interrupted job
uv run yt-dbl resume VIDEO_ID
```

## Commands
### `dub` — dub a video

```bash
uv run yt-dbl dub <URL> [options]
```

| Option | Description | Default |
|---|---|---|
| `-t`, `--target-language` | Target language | `ru` |
| `--bg-volume` | Background volume (0.0–1.0) | `0.15` |
| `--max-speed` | Max TTS speed-up (1.0–2.0) | `1.4` |
| `--max-models` | Max models in memory | auto (by RAM) |
| `--from-step` | Start from step: `download` / `separate` / `transcribe` / `translate` / `synthesize` / `assemble` | — |
| `--no-subs` | Disable subtitles | `false` |
| `--sub-mode` | Subtitle mode: `softsub` / `hardsub` / `none` | `softsub` |
| `--format` | Output format: `mp4` / `mkv` | `mp4` |

### `resume` — resume an interrupted job
```bash
uv run yt-dbl resume <video_id> [--max-models N]
```

The pipeline saves `state.json` after each step. If interrupted, `resume` picks up from the last incomplete step.

### `status` — check job status
```bash
uv run yt-dbl status <video_id>
```

Shows a table with each step's state (`pending` / `running` / `completed` / `failed`), execution time, and video metadata.

### `models list` — list ML models
```bash
uv run yt-dbl models list
```

Shows all models, their download status, and size on disk.

### `models download` — pre-download models
```bash
uv run yt-dbl models download
```

Downloads all HuggingFace models. The `audio-separator` model is downloaded automatically on first use.

## How it works
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                YouTube URL                                      │
└─────────────────────────────────────┬───────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  1. DOWNLOAD                                                                    │
│                                                                                 │
│  yt-dlp downloads the video, ffmpeg extracts the audio track                    │
│  Output: video.mp4, audio.wav (48 kHz, mono)                                    │
└─────────────────────────────────────┬───────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  2. SEPARATE                                                                    │
│                                                                                 │
│  BS-RoFormer splits audio into vocals and background (ONNX + CoreML)            │
│  Output: vocals.wav, background.wav                                             │
└──────────────────┬──────────────────────────────────────────┬───────────────────┘
                   │                                          │
              vocals.wav                                background.wav
                   │                                          │
                   ▼                                          │
┌──────────────────────────────────────────────────────┐      │
│  3. TRANSCRIBE                                       │      │
│                                                      │      │
│  VibeVoice-ASR (MLX, ~8 GB)                          │      │
│    → speech segments + speaker diarization           │      │
│  Qwen3-ForcedAligner (MLX, ~600 MB)                  │      │
│    → word-level timestamps                           │      │
│  + language auto-detection via Unicode scripts       │      │
│                                                      │      │
│  Output: segments.json                               │      │
└──────────────────────────┬───────────────────────────┘      │
                           │                                  │
                           ▼                                  │
┌──────────────────────────────────────────────────────┐      │
│  4. TRANSLATE                                        │      │
│                                                      │      │
│  Claude API (single-pass, all segments at once)      │      │
│  TTS-friendly output: short phrases, spelled-out     │      │
│  numbers, no special characters                      │      │
│                                                      │      │
│  Output: translations.json, subtitles.srt            │      │
└──────────────────────────┬───────────────────────────┘      │
                           │                                  │
                           ▼                                  │
┌──────────────────────────────────────────────────────┐      │
│  5. SYNTHESIZE                                       │      │
│                                                      │      │
│  Qwen3-TTS (MLX, ~1.7 GB) — voice cloning            │      │
│  using a voice reference for each speaker            │      │
│  Postprocessing (parallel, ThreadPool):              │      │
│    • speed-up (rubberband or atempo)                 │      │
│    • loudnorm (-16 LUFS, 2-pass)                     │      │
│    • de-essing                                       │      │
│                                                      │      │
│  Output: segment_0000.wav, segment_0001.wav ...      │      │
└──────────────────────────┬───────────────────────────┘      │
                           │                                  │
                           ▼                                  ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  6. ASSEMBLE                                                                    │
│                                                                                 │
│  Speech track (crossfade 50 ms, equal-power) + background (sidechain ducking)   │
│  + video (copy) + subtitles (softsub / hardsub / none)                          │
│  All in a single ffmpeg call                                                    │
│                                                                                 │
│  Output: result.mp4                                                             │
└─────────────────────────────────────┬───────────────────────────────────────────┘
                                      │
                                      ▼
                            ┌───────────────────┐
                            │    result.mp4     │
                            └───────────────────┘
```

### Memory management
ML models are loaded and unloaded via an LRU manager.

The number of models kept in memory is determined automatically based on available RAM:

```
RAM              Models     Batch (separation)
─────────────    ───────    ──────────────────
<= 16 GB         1          1
17–31 GB         2          2
32–47 GB         3          4
48+ GB           3          8
```

The ASR model (~8 GB) is unloaded before loading the Aligner so both don't occupy memory at the same time.

### Working directory structure
```
work/
└── <video_id>/
    ├── state.json                  ← pipeline checkpoint (JSON)
    ├── 01_download/
    │   ├── video.mp4               ← original video
    │   └── audio.wav               ← extracted audio track (48 kHz, mono)
    ├── 02_separate/
    │   ├── vocals.wav              ← isolated vocals
    │   └── background.wav          ← background music/noise
    ├── 03_transcribe/
    │   └── segments.json           ← segments, speakers, words with timestamps
    ├── 04_translate/
    │   ├── translations.json       ← translated texts
    │   └── subtitles.srt           ← subtitles (SRT)
    ├── 05_synthesize/
    │   ├── ref_SPEAKER_00.wav      ← speaker voice reference
    │   ├── segment_0000.wav        ← final segments (after postprocessing)
    │   ├── segment_0001.wav
    │   └── synth_meta.json         ← synthesis metadata
    └── 06_assemble/
        └── speech.wav              ← assembled speech track
    └── result.mp4                  ← final output (in job dir root)
```

## Models
| Model | Size | Task | Inference |
|---|---|---|---|
| [VibeVoice-ASR](https://huggingface.co/mlx-community/VibeVoice-ASR-bf16) | ~8.2 GB | ASR + speaker diarization | MLX (Metal) |
| [Qwen3-ForcedAligner](https://huggingface.co/mlx-community/Qwen3-ForcedAligner-0.6B-8bit) | ~600 MB | Word-level alignment | MLX (Metal) |
| [Qwen3-TTS](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16) | ~1.7 GB | TTS with voice cloning | MLX (Metal) |
| MelBand-RoFormer (BS-RoFormer) | ~200 MB | Vocal/background separation | ONNX + CoreML |
| Claude Sonnet 4.5 | — | Text translation | API (Anthropic) |

### Supported languages
**TTS (synthesis):** Russian, English, German, French, Spanish, Italian, Portuguese, Chinese, Japanese, Korean, Arabic, Hindi, Turkish, Dutch, Polish, Ukrainian

**ASR (recognition):** auto-detected via Unicode scripts (Latin, Cyrillic, Arabic, Devanagari, CJK, etc.)

## Development
```bash
just check          # lint + format + typecheck + tests
just test           # fast tests (parallel, coverage)
just test-e2e       # E2E tests (requires FFmpeg + network)
just fix            # auto-fix linter
just format         # auto-format
```

## License
MIT
