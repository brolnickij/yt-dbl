# CHANGELOG


## v1.0.0 (2026-02-06)

First stable release of yt-dbl — a CLI tool for dubbing YouTube videos using local ML models on Apple Silicon.

### Pipeline

- **Download** — extract audio from YouTube via yt-dlp
- **Separate** — isolate vocals/instrumentals with BS-RoFormer (SDR 12.97) via audio-separator
- **Transcribe** — ASR + diarization with VibeVoice-ASR (9B), word-level alignment with Qwen3-ForcedAligner (0.6B)
- **Translate** — single-pass translation via Claude Opus 4.6, TTS-optimized prompt (numbers→words, abbreviation expansion, stress marks for ru/uk)
- **Synthesize** — voice cloning TTS with Qwen3-TTS (1.7B), per-segment speed adjustment and loudness normalization
- **Assemble** — mix dubbed vocals with original instrumentals, sidechain ducking, equal-power crossfade at boundaries

### Audio Quality

- Rubberband pitch-preserving time-stretch (auto-detected ffmpeg-full)
- Two-pass EBU R128 loudness normalization (-16 LUFS)
- De-essing, highpass + denoise on voice references
- Sinc resampling via scipy to eliminate aliasing
- Confidence-weighted voice reference selection
- AAC output at 320 kbps

### Model Management

- LRU model manager with automatic load/unload
- RAM-based auto-detection of `max_loaded_models` and `separation_batch_size`
- Memory tracking with MLX/torch cache cleanup
- CLI commands: `yt-dbl models list`, `yt-dbl models download`
- Model registry with download status checks

### Performance

- Parallel segment postprocessing
- Load audio once, slice per segment for alignment
- Merged ffmpeg postprocessing into minimal-pass pipeline
- Aggressive memory cleanup between pipeline steps

### Developer Experience

- Rich progress bars for transcription and synthesis
- All configuration via environment variables (`YT_DBL_*`)
- Idempotent steps with JSON caching — resume from any point
- Structured working directory: `work/<video_id>/01_download/...06_assemble/`
- PEP 561 typed, strict mypy, ruff lint + format
- Unit / integration / E2E test pyramid
