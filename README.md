# yt-dbl
Dub any YouTube video into another language — with the original speaker's voice

```bash
yt-dbl dub "https://www.youtube.com/watch?v=VIDEO_ID" -t ru
```

> [!WARNING]
> **Early stage** — not yet stable for long videos (30+ min)

> [!WARNING]
> **Apple Silicon only** (M1–M4), tested on M4 Pro (48 GB)

One command: download, transcribe, translate (Claude), clone each speaker's voice (Qwen3-TTS), mix with the original background — done. All ML inference runs locally on your Mac's GPU via [MLX](https://github.com/ml-explore/mlx)


## Why yt-dbl
- **Human-quality voice cloning**<br>
Qwen3-TTS per speaker, not a generic synth. Multiple speakers are diarized and voiced separately
- **LLM translation**<br>
Claude handles idioms, context, and produces TTS-friendly text — not word-for-word machine translation
- **Background preserved**<br>
BS-RoFormer separates vocals from music/sfx. Sidechain ducking mixes them back naturally
- **Production audio chain**<br>
Loudnorm (-16 LUFS), de-essing, pitch-preserving speed-up, equal-power crossfade
- **Checkpoint & resume**<br>
Every step saves state. Interrupted? `yt-dbl resume` continues where it stopped
- **Private**<br>
Everything local except the Claude API call


## Supported languages
**TTS (synthesis):** Russian, English, German, French, Spanish, Italian, Portuguese, Chinese, Japanese, Korean, Arabic, Hindi, Turkish, Dutch, Polish, Ukrainian

**ASR (recognition):** auto-detected via Unicode scripts (Latin, Cyrillic, Arabic, Devanagari, CJK, etc.)


## Requirements
- **macOS** with **Apple Silicon** (M1–M4) — MLX needs Metal
- **Python** >= 3.12
- **FFmpeg** — audio extraction, postprocessing, final assembly
- **yt-dlp** — video download
- **Anthropic API key** — translation via Claude


## Installation
### 1. Install system dependencies
```bash
brew install ffmpeg yt-dlp
```

> **Optional:** `brew install ffmpeg-full` for pitch-preserving speed-up via rubberband
> Without it, falls back to ffmpeg's `atempo` filter (works fine, just no pitch correction)

### 2. Install yt-dbl
```bash
# From PyPI
uv tool install --prerelease=allow yt-dbl

# Or with pipx
pipx install yt-dbl
```

> `--prerelease=allow` is needed because `mlx-audio` depends on a pre-release `transformers`
>
> If `yt-dbl` is not found, run `uv tool update-shell && source ~/.zshrc`

<details>
<summary>From source</summary>

```bash
git clone git@github.com:brolnickij/yt-dbl.git && cd yt-dbl
uv sync
```

Use `uv run yt-dbl` instead of `yt-dbl` when running from source
</details>

### 3. Set up the API key
```bash
echo 'export YT_DBL_ANTHROPIC_API_KEY="sk-ant-..."' >> ~/.zshrc
source ~/.zshrc
```

Or use a `.env` file:

```env
YT_DBL_ANTHROPIC_API_KEY=sk-ant-...
```

### 4. Pre-download models (optional)
Models (~8.2 GB) download automatically on first run, or fetch them ahead of time:

```bash
yt-dbl models download
```


## Configuration
Priority: CLI args > env vars (`YT_DBL_` prefix) > `.env` file > defaults

```bash
cp .env.example .env
```

| Env variable | Default | Description |
|---|---|---|
| `YT_DBL_ANTHROPIC_API_KEY` | — | **Required** — Anthropic API key |
| `YT_DBL_TARGET_LANGUAGE` | `ru` | Target language (ISO 639-1) |
| `YT_DBL_OUTPUT_FORMAT` | `mp4` | `mp4` / `mkv` |
| `YT_DBL_SUBTITLE_MODE` | `softsub` | `softsub` / `hardsub` / `none` |
| `YT_DBL_BACKGROUND_VOLUME` | `0.15` | Background volume during speech (0.0–1.0) |
| `YT_DBL_MAX_SPEED_FACTOR` | `1.4` | Max TTS speed-up to fit timing (1.0–2.0) |
| `YT_DBL_MAX_LOADED_MODELS` | `0` (auto) | Max models in memory (0 = auto by RAM) |
| `YT_DBL_WORK_DIR` | `dubbed` | Output directory |

> See [`.env.example`](.env.example) for all 33 parameters


## Quick start
```bash
yt-dbl dub "https://www.youtube.com/watch?v=VIDEO_ID"           # dub to Russian (default)
yt-dbl dub "https://youtu.be/VIDEO_ID" -t es                    # dub to Spanish
yt-dbl dub "https://youtu.be/VIDEO_ID" -o ./out                 # custom output dir
yt-dbl dub "https://youtu.be/VIDEO_ID" --from-step translate    # re-run from a specific step
yt-dbl resume VIDEO_ID                                          # resume after interrupt
yt-dbl status VIDEO_ID                                          # check job progress
```


## Commands
### `dub` — dub a video
```bash
yt-dbl dub <URL> [options]
```

| Option | Description | Default |
|---|---|---|
| `-t`, `--target-language` | Target language | `ru` |
| `-o`, `--output-dir` | Output directory | `./dubbed` |
| `--bg-volume` | Background volume (0.0–1.0) | `0.15` |
| `--max-speed` | Max TTS speed-up (1.0–2.0) | `1.4` |
| `--max-models` | Max models in memory | auto |
| `--from-step` | Start from: `download` / `separate` / `transcribe` / `translate` / `synthesize` / `assemble` | — |
| `--no-subs` | Disable subtitles | `false` |
| `--sub-mode` | `softsub` / `hardsub` / `none` | `softsub` |
| `--format` | `mp4` / `mkv` | `mp4` |

### `resume` — pick up where it stopped
```bash
yt-dbl resume <video_id> [--max-models N] [-o DIR]
```

### `status` — check job progress
```bash
yt-dbl status <video_id>
```

### `models list` / `models download`
```bash
yt-dbl models list        # show models, download status, size
yt-dbl models download    # pre-download all models
```


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
└───────────────────────────┬────────────────────────────────────────────┬────────┘
                            │                                            │
                       vocals.wav                                  background.wav
                            │                                            │
                            ▼                                            │
┌──────────────────────────────────────────────────────┐                 │
│  3. TRANSCRIBE                                       │                 │
│                                                      │                 │
│  VibeVoice-ASR (MLX, ~5.7 GB)                        │                 │
│    → speech segments + speaker diarization           │                 │
│  Qwen3-ForcedAligner (MLX, ~600 MB)                  │                 │
│    → word-level timestamps                           │                 │
│  + language auto-detection via Unicode scripts       │                 │
│                                                      │                 │
│  Output: segments.json                               │                 │
└──────────────────────────┬───────────────────────────┘                 │
                           │                                             │
                           ▼                                             │
┌──────────────────────────────────────────────────────┐                 │
│  4. TRANSLATE                                        │                 │
│                                                      │                 │
│  Claude API (single-pass, all segments at once)      │                 │
│  TTS-friendly output: short phrases, spelled-out     │                 │
│  numbers, no special characters                      │                 │
│                                                      │                 │
│  Output: translations.json, subtitles.srt            │                 │
└──────────────────────────┬───────────────────────────┘                 │
                           │                                             │
                           ▼                                             │
┌──────────────────────────────────────────────────────┐                 │
│  5. SYNTHESIZE                                       │                 │
│                                                      │                 │
│  Qwen3-TTS (MLX, ~1.7 GB) — voice cloning            │                 │
│  using a voice reference for each speaker            │                 │
│  Postprocessing (parallel, ThreadPool):              │                 │
│    • speed-up (rubberband or atempo)                 │                 │
│    • loudnorm (-16 LUFS, 2-pass)                     │                 │
│    • de-essing                                       │                 │
│                                                      │                 │
│  Output: segment_0000.wav, segment_0001.wav ...      │                 │
└──────────────────────────┬───────────────────────────┘                 │
                           │                                             │
                           ▼                                             ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  6. ASSEMBLE                                                                    │
│                                                                                 │
│  Speech track (crossfade 50 ms, equal-power) + background (sidechain ducking)   │
│  + video (copy) + subtitles (softsub / hardsub / none)                          │
│  All in a single ffmpeg call                                                    │
│                                                                                 │
│  Output: result.mp4                                                             │
└──────────────────────────────────────────┬──────────────────────────────────────┘
                                           │
                                           ▼
                                 ┌───────────────────┐
                                 │    result.mp4     │
                                 └───────────────────┘
```

### Memory management
LRU model manager — auto-selects how many models to keep loaded based on RAM:

```
RAM              Models     Batch (separation)
─────────────    ───────    ──────────────────
<= 16 GB         1          1
17–31 GB         2          2
32–47 GB         3          4
48+ GB           3          8
```

ASR (~5.7 GB) is unloaded before loading the Aligner to avoid holding both in memory

### Output directory structure
```
dubbed/
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
    ├── 06_assemble/
    │   └── speech.wav              ← assembled speech track
    └── result.mp4                  ← final output (in job dir root)
```


## Models
| Model | Size | Task |
|---|---|---|
| [VibeVoice-ASR](https://huggingface.co/mlx-community/VibeVoice-ASR-4bit) | ~5.7 GB | ASR + speaker diarization |
| [Qwen3-ForcedAligner](https://huggingface.co/mlx-community/Qwen3-ForcedAligner-0.6B-8bit) | ~600 MB | Word-level alignment |
| [Qwen3-TTS](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16) | ~1.7 GB | TTS + voice cloning |
| MelBand-RoFormer (BS-RoFormer) | ~200 MB | Vocal/background separation |
| Claude Sonnet 4.5 | — | Translation (API) |

All local models run on MLX (Metal GPU), total ~8.2 GB


## Development
```bash
just check    # lint + format + typecheck + tests
just test     # fast tests (parallel, coverage)
just test-e2e # E2E (needs ffmpeg + network)
just fix      # auto-fix lint
just format   # auto-format
```


## License
MIT
