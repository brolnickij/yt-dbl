# CHANGELOG


## v1.5.0 (2026-02-06)

### Bug Fixes

- **synthesize**: Handle thread pool errors in postprocessing gracefully
  ([`455da3d`](https://github.com/brolnickij/yt-dbl/commit/455da3d19e655f86df690140d66519efeb5b3f35))

### Chores

- Bump uv.lock
  ([`67f7279`](https://github.com/brolnickij/yt-dbl/commit/67f727980ec8cf3d3eaa2f6856f095df2a4f49a6))

### Documentation

- **config**: Explain object.__setattr__ in model_validator
  ([`6194bef`](https://github.com/brolnickij/yt-dbl/commit/6194bef75bcbd62e0957df931012c7848a2c2582))

### Features

- **synthesize**: Add per-segment TTS retry with configurable max retries
  ([`4df96ef`](https://github.com/brolnickij/yt-dbl/commit/4df96ef6e9b7b25999c2fd76d84ff79ce4de419d))

### Refactoring

- **models**: Add Protocol types for ML models instead of Any
  ([`1e9de13`](https://github.com/brolnickij/yt-dbl/commit/1e9de134fcad7f1380db7ec24a4e2e54d4c33bba))

- **transcribe**: Use Hungarian algorithm for speaker reconciliation
  ([`a14268c`](https://github.com/brolnickij/yt-dbl/commit/a14268cea185334b8bc8c9bfdcf5ce33b27b49d2))

- **translate**: Extract system prompt to external template file
  ([`2c63266`](https://github.com/brolnickij/yt-dbl/commit/2c632663866efa2fb1c3d2c553f25cd72d1b1429))

### Testing

- Ignore type checking for lambda in model registration
  ([`6419919`](https://github.com/brolnickij/yt-dbl/commit/64199199d4bcd7cabcca59cde795c9be9d049903))


## v1.4.1 (2026-02-06)

### Bug Fixes

- Apply --target-language override to existing jobs instead of silently ignoring it
  ([`03dfeec`](https://github.com/brolnickij/yt-dbl/commit/03dfeeca1eb708321ecc5d7acaead85dc3863691))

- Use atomic write for state.json to prevent corruption on crash
  ([`9b06ae0`](https://github.com/brolnickij/yt-dbl/commit/9b06ae0e615c97b04f54a1985e853f71d5cf232f))

- **download**: Kill yt-dlp subprocess on exception to prevent zombies
  ([`ca5bc88`](https://github.com/brolnickij/yt-dbl/commit/ca5bc88f8d70420b737171b4e5e9ce56eed029ed))

- **runner**: Unload models on KeyboardInterrupt via try/finally
  ([`2e374a2`](https://github.com/brolnickij/yt-dbl/commit/2e374a270b5be9ec29da43d31c2f95ec722fc09b))

- **transcribe**: Persist source_language in segments.json
  ([`188a9c8`](https://github.com/brolnickij/yt-dbl/commit/188a9c81672123bf4f045c4a219f031c21b9e6b2))

### Documentation

- Init AGENTS.md
  ([`8a79ddd`](https://github.com/brolnickij/yt-dbl/commit/8a79dddbecabffaeb49bbf94f1ac45e6af0828aa))


## v1.4.0 (2026-02-06)

### Bug Fixes

- **translate**: Bump default max_tokens to 32768 for Sonnet 4.5
  ([`deba054`](https://github.com/brolnickij/yt-dbl/commit/deba054781853e7196c70c6f06f46e19fef197ac))

### Features

- **translate**: Add auto-batching for long audio translation
  ([`cc78e15`](https://github.com/brolnickij/yt-dbl/commit/cc78e15f3948577ce0a6e397961c3dd7ae75edb3))

### Testing

- Fix translation step mock and GPU memory cleanup tests
  ([`8a83e66`](https://github.com/brolnickij/yt-dbl/commit/8a83e665a6b08212a6dadbb0e1d2ed15bf5891dd))


## v1.3.0 (2026-02-06)

### Documentation

- Add --prerelease=allow to uv tool install instructions
  ([`9d8466a`](https://github.com/brolnickij/yt-dbl/commit/9d8466a7b818947095fdcd07bebe875bc3a4c1e7))

- Add PATH hint for uv tool install
  ([`aba2733`](https://github.com/brolnickij/yt-dbl/commit/aba27337f8b2709e0a7b2c5c3c04d71dfebd9e9a))

- Expand .env.example with all settings and update README config section
  ([`1ed55e7`](https://github.com/brolnickij/yt-dbl/commit/1ed55e70c814dcd22b6dfe7a64563a2baa4d275d))

- **readme**: Trim config table to key params, link to .env.example
  ([`8ae79cd`](https://github.com/brolnickij/yt-dbl/commit/8ae79cda5dc15d3c022b032acb83a685ef3cbcd0))

### Features

- **cli**: Disable built-in completion options
  ([`5ee2897`](https://github.com/brolnickij/yt-dbl/commit/5ee28974d16eb1f0a39478a96f5e1a41984c5300))

- **transcribe**: Add chunked ASR for audio longer than 55 minutes
  ([`ba010ce`](https://github.com/brolnickij/yt-dbl/commit/ba010ce011a3add4d46ab7988d538954f6bb4aa8))


## v1.2.0 (2026-02-06)

### Bug Fixes

- Add explicit transformers pre-release dep for uv tool install compatibility
  ([`88375fc`](https://github.com/brolnickij/yt-dbl/commit/88375fcc323b8e4470ec7d84b49970eba50cc066))

### Features

- **config**: Add separation_use_autocast setting for FP16 mixed precision
  ([`06caead`](https://github.com/brolnickij/yt-dbl/commit/06caead7c8f9a59cf54ba7f7da99bdbfca623264))

### Performance Improvements

- **separate**: Enable FP16 autocast and add inference timing
  ([`f4d0cce`](https://github.com/brolnickij/yt-dbl/commit/f4d0ccec6af0972a8e446d42b2ea54395cd141c1))

- **transcribe**: Switch VibeVoice-ASR from bf16 to 4bit quantization
  ([`9101509`](https://github.com/brolnickij/yt-dbl/commit/9101509329a5a1fb795936f324e6d3dcea0993ea))

### Testing

- **separate**: Add tests for autocast config option
  ([`3d0d75e`](https://github.com/brolnickij/yt-dbl/commit/3d0d75e07d1b055f2e7fc04f2f0b56376b06a723))


## v1.1.0 (2026-02-06)

### Chores

- Cleanup changelog
  ([`7cdc6d4`](https://github.com/brolnickij/yt-dbl/commit/7cdc6d479206ee0c9881f47e85d246df8231128a))

### Documentation

- Cleanup
  ([`1d71a65`](https://github.com/brolnickij/yt-dbl/commit/1d71a650cf65d2d6bfcb295b077eecb4ff00b231))

### Features

- Update output directory to `dubbed` & add output dir option `-o` in CLI
  ([`88fbca9`](https://github.com/brolnickij/yt-dbl/commit/88fbca9d5fdf063b5fee40958e9468a4e051db25))


## v1.0.0 (2026-02-06)

### Bug Fixes

- Add missing __init__.py to models package
  ([`e4c4b5b`](https://github.com/brolnickij/yt-dbl/commit/e4c4b5be5ab1a9b7ebeef2a205aa1facfd0d5c37))

- Add sidechain ducking and limiter to reduce robotic artifacts in mixed audio
  ([`ac60ce0`](https://github.com/brolnickij/yt-dbl/commit/ac60ce081a49693dd56514c2ae2a4e86b0a50f2d))

- Allow lazy torch import in model manager lint config
  ([`110a3ca`](https://github.com/brolnickij/yt-dbl/commit/110a3ca23573a55e72decb68d69a4ebb54921c13))

- Correct TTS sample rate from 12kHz to 24kHz (critical audio fix)
  ([`693e5db`](https://github.com/brolnickij/yt-dbl/commit/693e5dbdbf4643b12a657f5013ffd84524a0f409))

The Qwen3-TTS model outputs audio at 24000 Hz (model.sample_rate), but we were writing WAV files at
  12000 Hz. This caused all synthesized audio to play at half speed with pitched-down 'morphed'
  voices.

Also: - Revert TTS params to model defaults (temp=0.9, top_k=50, top_p=1.0) - Remove dead speed
  pre-calculation logic (model.generate speed param is documented as 'not directly supported yet') -
  Update tests to match corrected sample rate and defaults

- Improve TTS quality with proper Qwen3-TTS parameters
  ([`73a9cbb`](https://github.com/brolnickij/yt-dbl/commit/73a9cbb21631b55479bff407102003bc9f2b299e))

- Set temperature to 0.6 (model default) instead of 0.9 - Add top_k=20, top_p=0.8,
  repetition_penalty=1.05 params - Use native 'speed' param in model.generate() to pre-fit
  translated text into original segment duration, reducing reliance on quality-degrading post-hoc
  ffmpeg atempo - Add _estimate_speaking_rate() and _estimate_tts_speed() methods for intelligent
  speed pre-calculation - Add unit tests for new config defaults and speed estimation

- Resolve relative paths in _rename_outputs, add E2E tests
  ([`7edc5f8`](https://github.com/brolnickij/yt-dbl/commit/7edc5f87f2d8bc20620be0a094889371ec165818))

- Fix _rename_outputs to resolve relative paths against step_dir (audio-separator returns filenames,
  not absolute paths) - Add test_e2e_separate.py: 4 E2E tests for real BS-RoFormer separation - Add
  unit test for relative path handling in _rename_outputs - Update E2E pipeline tests: require
  audio_separator_available fixture - CI: restrict test job to tests/unit/, remove e2e job -
  Justfile: run E2E with -n0 to prevent concurrent model downloads

All checks pass: lint ✓ format ✓ mypy ✓ 115 unit/integration ✓ 10 E2E ✓

- Strip ANSI codes in CLI help test for narrow CI terminals
  ([`b4ac57b`](https://github.com/brolnickij/yt-dbl/commit/b4ac57b3f0055ddd58f015f290d00289e8185d16))

- Update environment variable names for clarity and consistency
  ([`fdfa808`](https://github.com/brolnickij/yt-dbl/commit/fdfa8081583bad2b796522fee69ba18e2173b509))

- **release**: Use python -m build in semantic-release Docker container
  ([`0ad5354`](https://github.com/brolnickij/yt-dbl/commit/0ad5354a3112c02ec4ff0b8b72a8f3cdd87c388b))

### Build System

- Add conventional commits enforcement via pre-commit
  ([`fbcd057`](https://github.com/brolnickij/yt-dbl/commit/fbcd057c245395cec621c8a20618a866d441364c))

- Add justfile for development tasks
  ([`3a056f8`](https://github.com/brolnickij/yt-dbl/commit/3a056f8543b0b2817a8c17e3229cf4d8632ff8fc))

- Bump all dependencies to latest versions
  ([`733472a`](https://github.com/brolnickij/yt-dbl/commit/733472a11f95630c8422399e524312e6d4fac8f6))

- Bump all dependency minimum versions to latest
  ([`ce66a82`](https://github.com/brolnickij/yt-dbl/commit/ce66a82257164c2e50226b2507f85b8ff9f54cdd))

- anthropic: >=0.52 -> >=0.78 - rich: >=14.0 -> >=14.3.2 - pydantic: >=2.12 -> >=2.12.5 - yt-dlp:
  >=2026.0 -> >=2026.2 - audio-separator: >=0.41 -> >=0.41.1 - mlx-audio: >=0.3 -> >=0.3.1 - typer:
  >=0.21 -> >=0.21.1 - mypy: >=1.19 -> >=1.19.1 - pytest: >=9.0 -> >=9.0.2 - pytest-randomly: >=4.0
  -> >=4.0.1 - pre-commit: >=4.5 -> >=4.5.1 - pre-commit hooks additional_dependencies synced

- Update lockfile with new dependencies
  ([`53a9ca8`](https://github.com/brolnickij/yt-dbl/commit/53a9ca82409c706193ea10a83be494fa37d9a614))

### Chores

- Add TC002 to test per-file-ignores in ruff config
  ([`ecef19b`](https://github.com/brolnickij/yt-dbl/commit/ecef19b52c0594b1ef17332354e848f061f0366a))

- Enhance pyproject.toml
  ([`aedd66f`](https://github.com/brolnickij/yt-dbl/commit/aedd66f56a823be10e41da1362feb5b11025905e))

- Remove unused network marker and dead _HAS_MLX variable
  ([`c7e00c2`](https://github.com/brolnickij/yt-dbl/commit/c7e00c26cb41340bb9927f463d3747ddef3f093d))

- Stable version
  ([`0518d3e`](https://github.com/brolnickij/yt-dbl/commit/0518d3eee2339fdb710eac6e11b464c86f939075))

- Update .gitignore with coverage and mypy cache
  ([`7325c43`](https://github.com/brolnickij/yt-dbl/commit/7325c4309f88a2891f0de64cd78af52ed981454f))

### Code Style

- Use parameterless raise for typer.Exit
  ([`51e8ca7`](https://github.com/brolnickij/yt-dbl/commit/51e8ca73b33a796943ab1909a3eefae5684a776d))

### Continuous Integration

- Add GitHub Actions workflow for lint, typecheck, and tests
  ([`cf3f0d3`](https://github.com/brolnickij/yt-dbl/commit/cf3f0d348292236fac3ab117ac74a8b0a0139f5a))

- Add platform marker for mlx-audio, set UV_TORCH_BACKEND=cpu in CI
  ([`ed01b02`](https://github.com/brolnickij/yt-dbl/commit/ed01b022b1fe33d9456b292ef7d513bfb13a973e))

- Drop Python version matrix in favor of single 3.12
  ([`b9b7063`](https://github.com/brolnickij/yt-dbl/commit/b9b7063174e635ece70f1fe7282c3914f336cf7e))

- Force E2E tests to run sequentially (xdist_group)
  ([`bdd201f`](https://github.com/brolnickij/yt-dbl/commit/bdd201fcf4a55cb3cc0f9641d387168ac9e61ffc))

GPU-heavy models (VibeVoice-ASR 9B, BS-RoFormer) kill the machine when loaded in parallel. All E2E
  tests now share a single xdist group via conftest hook + --dist loadgroup in pytest.ini.

- Include integration tests
  ([`fb72eee`](https://github.com/brolnickij/yt-dbl/commit/fb72eee65eb39ec9d6b8baf26779b2f5964c1f16))

### Documentation

- Bump
  ([`0812074`](https://github.com/brolnickij/yt-dbl/commit/081207411751992b400907a09b86713b0f081e27))

- Bump
  ([`10c1d62`](https://github.com/brolnickij/yt-dbl/commit/10c1d62a8edf9e28c55e1eda90a342292492b344))

- Bump readme.md
  ([`1ad90a2`](https://github.com/brolnickij/yt-dbl/commit/1ad90a2429ce33193eccc4904b67a3252c51792c))

- Bump readme.md
  ([`e2679d7`](https://github.com/brolnickij/yt-dbl/commit/e2679d775790e6f9d79ae37dd93203e9ed4f4b13))

- Bump readme.md
  ([`e8cf40e`](https://github.com/brolnickij/yt-dbl/commit/e8cf40e9bf3919e850f25c92e114359dfb2424ff))

- Cleanup
  ([`baf712e`](https://github.com/brolnickij/yt-dbl/commit/baf712ebe2da3c851ed7dd05ce38792da0bb5a40))

- Cleanup
  ([`a87505a`](https://github.com/brolnickij/yt-dbl/commit/a87505a6a3fb8d042633173da9fab3b17b055242))

- Fix repository URL
  ([`392b214`](https://github.com/brolnickij/yt-dbl/commit/392b214db46bce01fb433732e8d3c6d7235a41dd))

- Improve formatting and clarity in README.md
  ([`0cd95a7`](https://github.com/brolnickij/yt-dbl/commit/0cd95a7e3d709dacabd695678d96ea835edbfb8c))

- Update README
  ([`eb2e9ab`](https://github.com/brolnickij/yt-dbl/commit/eb2e9ab142c3cc7ce2459a91d1b3d3988c1a90d0))

- Update requirements & install
  ([`9d1d826`](https://github.com/brolnickij/yt-dbl/commit/9d1d8261d74f2ec57fcc4ddaa26f58a7c5f1117e))

- **separate**: Document ModelManager exclusion for separator models
  ([`c15487f`](https://github.com/brolnickij/yt-dbl/commit/c15487fd9edc0cd1f1a8517813af57031e114de1))

### Features

- Add LRU model manager
  ([`b577721`](https://github.com/brolnickij/yt-dbl/commit/b577721e9a83d7696f3b909206dfde90cf57ea2f))

- Add Rich progress bars to transcribe and synthesize steps
  ([`ebb2d43`](https://github.com/brolnickij/yt-dbl/commit/ebb2d43bdc456c7def2a857ea02a969f56950f60))

- ASR: animated spinner with status during long model.generate() call - Alignment: progress bar
  showing segment N/M - TTS synthesis: progress bar per segment - Postprocessing: progress bar for
  speed/loudnorm

- Add stress marks to Russian translations for better TTS pronunciation
  ([`c6b9683`](https://github.com/brolnickij/yt-dbl/commit/c6b9683d3b7f1f97e0ec107f85d66d816bc0e240))

- Add U+0301 combining acute accent instruction to Claude translation prompt for Russian (ru) and
  Ukrainian (uk) target languages - Stress marks affect Qwen3-TTS tokenization, improving word
  stress and pronunciation naturalness - Disable ruff RUF001/RUF003 (ambiguous Unicode) — Cyrillic
  text is intentional - Zero extra dependencies or API calls (reuses existing Claude translation
  call)

- Confidence-weighted voice reference selection for better cloning
  ([`27f9738`](https://github.com/brolnickij/yt-dbl/commit/27f97389331f1343cc27233121230761d953d2ec))

- De-ess sibilants in TTS postprocessing chain
  ([`334bb09`](https://github.com/brolnickij/yt-dbl/commit/334bb0959ad3ce1394e0e4f31422ddd627cedd2c))

- Enhance TTS guidelines for better pronunciation and clarity
  ([`6688b45`](https://github.com/brolnickij/yt-dbl/commit/6688b4520eb27b0e211f8ede72d40b4d2d090c35))

- Equal-power crossfade for constant energy at segment boundaries
  ([`03ed67a`](https://github.com/brolnickij/yt-dbl/commit/03ed67a3825a262b67e2221b4892886a1ea04361))

- Highpass + denoise filter on voice reference for cleaner cloning
  ([`393815f`](https://github.com/brolnickij/yt-dbl/commit/393815fee7e43d1cef0eb9b080eba92b51b0d642))

- Implement real vocal separation with BS-RoFormer
  ([`8c394f0`](https://github.com/brolnickij/yt-dbl/commit/8c394f0ee7bb85f4bbb0842f8d031d1ca49b205e))

- Replace stub SeparateStep with audio-separator integration - Use BS-RoFormer (SDR 12.97) as
  default model - CoreML acceleration for Apple Silicon (M4 Pro) - Configurable model, segment_size,
  overlap via Settings - Idempotency: skip if outputs already exist - Memory cleanup: gc.collect() +
  GPU cache clear after separation - 17 new unit tests for validation, rename, config, error paths -
  Updated pipeline integration tests with mock separation - Added prefill_separate helper to
  conftest

- Implement real yt-dlp download step with audio extraction
  ([`2609176`](https://github.com/brolnickij/yt-dbl/commit/260917670e2fba86b3c72ef793b5ea2be0de9585))

- Implement single-pass translation via Claude Opus 4.6 (iter 4)
  ([`b78ed00`](https://github.com/brolnickij/yt-dbl/commit/b78ed004c20d7a85b32ff1276d29a3441fb13485))

- Add anthropic>=0.52 runtime dependency - TranslateStep: single API call with all segments, JSON
  output - System prompt for dubbing-aware translation (concise for TTS timing) - Idempotent: caches
  translations.json, regenerates subtitles.srt - Parse Claude response with markdown fence handling
  - Config: claude_model='claude-opus-4-6', removed translation_batch_size - 22 unit tests
  (validation, parsing, SRT, persistence, mocked API) - Integration tests updated with
  _pipeline_patches helper - 168 tests, 91.92% coverage, all quality checks pass

- Implement transcription with VibeVoice-ASR + ForcedAligner (iter 3)
  ([`decd55c`](https://github.com/brolnickij/yt-dbl/commit/decd55c08ffec0c327d68dd5123e8329842f1275))

Two-model pipeline running on Apple Silicon MLX Metal: 1. VibeVoice-ASR (9B) — ASR + diarization +
  segment timestamps 2. Qwen3-ForcedAligner (0.6B) — word-level forced alignment

Key changes: - Add mlx-audio>=0.3 dependency (requires prerelease transformers) - Add transcription
  config: asr_model, aligner_model, max_tokens, temperature - Rewrite TranscribeStep from stub to
  full implementation: - _run_asr(): loads VibeVoice, runs generate(), normalises key variants -
  _run_alignment(): per-segment forced alignment with error fallback - _detect_language(): Unicode
  heuristics (ja before zh for kana) - _extract_speakers(): duration aggregation + best reference
  segment - Idempotent with JSON cache, gc.collect() between models - 146 tests (25 new for
  transcription), 90.93% coverage - All quality checks pass: ruff lint, ruff format, mypy strict

- Implement TTS synthesis with Qwen3-TTS voice cloning (iter 5)
  ([`c4330e5`](https://github.com/brolnickij/yt-dbl/commit/c4330e528fa25489abc6277e574f0fc319d5760a))

- Implement SynthesizeStep: voice reference extraction, TTS generation, speed adjustment (ffmpeg
  atempo), loudness normalization (-16 LUFS) - Model: mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16
  via mlx-audio - Voice cloning via ref_audio + ref_text params - Add tts_model, tts_temperature,
  tts_sample_rate to Settings - Add synth_path, synth_speed_factor to Segment schema - Add
  soundfile>=0.13 dependency for WAV writing - Add 20 unit tests for synthesize step (all passing) -
  Update integration tests with TTS mocks - Add prefill_synthesize helper to conftest - 188 tests,
  91.14% coverage

- Init
  ([`68eb169`](https://github.com/brolnickij/yt-dbl/commit/68eb169be5ab510a00d1db0913f20085d89219d1))

- Init assemble module
  ([`e88b499`](https://github.com/brolnickij/yt-dbl/commit/e88b499c4e4ec148cae228958b431f762ecbc01e))

- Init publish configs
  ([`dee5cbf`](https://github.com/brolnickij/yt-dbl/commit/dee5cbfa0de6c3b4b94787b439c4cb0b6eca6b70))

- Initial public release
  ([`d112074`](https://github.com/brolnickij/yt-dbl/commit/d112074bd781c73c438380dd352ab7010a6f0761))

BREAKING CHANGE: first stable release of yt-dbl

- Make separation batch_size configurable via env
  ([`396ca6c`](https://github.com/brolnickij/yt-dbl/commit/396ca6c31f9570455d8028e3c73054fda08f5a6d))

Add YT_DBL_SEPARATION_BATCH_SIZE (default=1, max=16). On M4 Pro with unified memory, higher values
  (2-4) can speed up separation.

- Optimize translation prompt for TTS-friendly output
  ([`fe50b79`](https://github.com/brolnickij/yt-dbl/commit/fe50b798137a5ed75140115c284c18a5768f76b5))

- Add rules: numbers→words, expand abbreviations, spoken style, no special chars - Fix rule
  numbering bug (duplicate '7.' when stress rule injected) - Add stress-marked example in JSON
  output for ru/uk languages - Rephrase prompt intro to emphasize TTS optimization context

- Sinc resampling via scipy to eliminate aliasing artifacts
  ([`9b9886f`](https://github.com/brolnickij/yt-dbl/commit/9b9886f52e437ae3be36a120525c040d501b628e))

- Suppress HuggingFace/transformers noise during model loading
  ([`34671fa`](https://github.com/brolnickij/yt-dbl/commit/34671fa191aac3cca65a32e9ee249b68cb98a34d))

- Add suppress_library_noise() context manager to logging utils - Silence HF Hub download progress
  bars (HF_HUB_DISABLE_PROGRESS_BARS) - Suppress 'model of type' and tokenizer regex warnings - Mute
  noisy loggers: huggingface_hub, transformers, tqdm, etc. - Wrap all 3 model loads: ASR, aligner,
  TTS

- Update TTS mock functions for improved testing and remove mlx dependency
  ([`b929fa8`](https://github.com/brolnickij/yt-dbl/commit/b929fa891b841fcfdeb0f4b51518a0bda315d360))

- Use ffmpeg-full with rubberband for higher quality audio
  ([`feac9a9`](https://github.com/brolnickij/yt-dbl/commit/feac9a9662864158848af40592d2c62835477fb6))

- Auto-detect ffmpeg-full (brew keg-only) with fallback to system ffmpeg - Rubberband
  pitch-preserving time-stretch instead of atempo - Two-pass loudnorm (measure → apply with
  linear=true) - Voice references at 24kHz (Qwen3-TTS native rate, was 16kHz) - AAC output at
  320kbps (was default ~128kbps) - Add ffmpeg_path config setting for custom ffmpeg location - Tests
  for _detect_ffmpeg, _detect_ffprobe, has_rubberband

- **cli**: Implement real models list and download commands
  ([`b336cbf`](https://github.com/brolnickij/yt-dbl/commit/b336cbfed7f4edf94ea47e30bf2b5445d857f412))

- **config**: Auto-detect max_loaded_models based on system RAM
  ([`38ec78f`](https://github.com/brolnickij/yt-dbl/commit/38ec78f2122e55e6466bce203a8627bf08052570))

- **logging**: Add memory measurement and enhanced model load/unload logging
  ([`7db825b`](https://github.com/brolnickij/yt-dbl/commit/7db825bc41c07af2b166d6f29ad527abe8aa6686))

- **manager**: Add memory tracking and MLX/torch cache cleanup
  ([`9d3f746`](https://github.com/brolnickij/yt-dbl/commit/9d3f746db319a15cf763530b5627ee35ad3d1f96))

- **models**: Export public API from models package
  ([`5893660`](https://github.com/brolnickij/yt-dbl/commit/5893660e5df255ee25629aa6336cf1435b9a6539))

- **pipeline**: Accept ModelManager in PipelineStep base class
  ([`740dd96`](https://github.com/brolnickij/yt-dbl/commit/740dd96b33bce7ad07a66ed4343a30886099cb47))

- **registry**: Add ML model registry with download status checks
  ([`2a3e8f1`](https://github.com/brolnickij/yt-dbl/commit/2a3e8f1d1332967c34d8db5c0e077f52b24f3217))

- **runner**: Wire ModelManager and memory status into pipeline runner
  ([`34795ac`](https://github.com/brolnickij/yt-dbl/commit/34795ac7ad5b600a2fbc66f185ccdbb94f24339e))

- **schemas**: Add source_language field to PipelineState
  ([`e00b27a`](https://github.com/brolnickij/yt-dbl/commit/e00b27ab871f677c7f683e5cb96327efbc8bf838))

- **synthesize**: Integrate ModelManager for TTS model lifecycle
  ([`0cca551`](https://github.com/brolnickij/yt-dbl/commit/0cca551451173f82c81d6766b33250a3fca304d7))

- **transcribe**: Integrate ModelManager and expand language detection
  ([`2029b71`](https://github.com/brolnickij/yt-dbl/commit/2029b7109e5048ed791b5171bd4be48860e718bd))

- **translate**: Pass source language to Claude translation prompt
  ([`93376bb`](https://github.com/brolnickij/yt-dbl/commit/93376bb40dc8824629e1c9be7fd3def22edef82d))

### Performance Improvements

- Cache has_rubberband, unload ASR before aligner, precompute lookups, compact JSON, cleanup
  intermediates
  ([`8422144`](https://github.com/brolnickij/yt-dbl/commit/842214483d4ab75c25e9a73718c0e3dc6d1e36c2))

- Enhance RAM-based auto-detection for `max_loaded_models` and `separation_batch_size`
  ([`94f479a`](https://github.com/brolnickij/yt-dbl/commit/94f479af98612efed509ddfeced8bd96bfb44ab1))

- Enhance segment postprocessing with parallel execution
  ([`9d1e296`](https://github.com/brolnickij/yt-dbl/commit/9d1e2962a6568dba7b9f6dc088f5410d5e3a4f0c))

- **synthesize**: Merge ffmpeg postprocessing into minimal-pass pipeline
  ([`6db80ef`](https://github.com/brolnickij/yt-dbl/commit/6db80ef75ec61140923d3462c96cf85b2dc51521))

- **synthesize**: Use np.asarray instead of tolist for mlx-to-numpy conversion
  ([`c6d2bc4`](https://github.com/brolnickij/yt-dbl/commit/c6d2bc469bef586ada386add7d4fb0af6b0ae3f8))

- **transcribe**: Load audio once and slice per segment for alignment
  ([`9586e6d`](https://github.com/brolnickij/yt-dbl/commit/9586e6d9f0ef146d3e99c3b24c75fc08e9078022))

### Refactoring

- Add __all__ exports to public modules
  ([`aff728c`](https://github.com/brolnickij/yt-dbl/commit/aff728c6b5fdbd740821e61d859853e960fbce27))

- Add PEP 561 py.typed marker
  ([`ac757cc`](https://github.com/brolnickij/yt-dbl/commit/ac757cc30cf24aac805221a74470172543cbed99))

- Complete pipeline error hierarchy with TranscriptionError and AssemblyError wrapping
  ([`4160359`](https://github.com/brolnickij/yt-dbl/commit/416035944b8d3a260a60de31839ed62ac7710fe8))

- Deduplicate atempo chain in speed_up_audio
  ([`4750e44`](https://github.com/brolnickij/yt-dbl/commit/4750e442d55153410cba53411991320693dffa65))

- Deduplicate resolve_vocals and GPU cleanup into shared utilities
  ([`ce1be33`](https://github.com/brolnickij/yt-dbl/commit/ce1be3399e521f104c2e4e7e9fc267cbe6bc9d32))

- Extract audio processing utilities from SynthesizeStep into utils/audio_processing
  ([`37ae8d1`](https://github.com/brolnickij/yt-dbl/commit/37ae8d113798009249f95fa780259bab3136c5bb))

- Extract constants, fix inconsistencies, deduplicate loudnorm targets
  ([`16c815d`](https://github.com/brolnickij/yt-dbl/commit/16c815dc4a0ccd08a3f7bd7c14e5179457342e84))

- Extract language maps, lazy imports, early validation, remove dead code
  ([`a193791`](https://github.com/brolnickij/yt-dbl/commit/a19379117a5b62f09fa5d122979006e86045f621))

- Extract tool configs into standalone files
  ([`cd98bf8`](https://github.com/brolnickij/yt-dbl/commit/cd98bf818a2d910335bfa9119cf95bb1cc9dac8d))

- Fix type safety in assemble, promote private registry API to public
  ([`e9c7c68`](https://github.com/brolnickij/yt-dbl/commit/e9c7c68a208b3057fc388e92da41f81374569553))

- Inject ffmpeg path via set_ffmpeg_path instead of importing settings singleton
  ([`eba5647`](https://github.com/brolnickij/yt-dbl/commit/eba564712f7193d6a076f29af473cdecc2f2d3bb))

- Introduce unified pipeline exception hierarchy
  ([`4392987`](https://github.com/brolnickij/yt-dbl/commit/4392987b6ad6045a3cb4183cb527c7fbce1f4c09))

- Remove commented-out code
  ([`489eeb2`](https://github.com/brolnickij/yt-dbl/commit/489eeb2c946148c0e8a39ec8380ca4b19f73ff95))

- Remove dead code (replace_audio, normalize_loudness, deess, TranscriptionError)
  ([`e8dbc81`](https://github.com/brolnickij/yt-dbl/commit/e8dbc81ac70eb3e240ac8fce3d8c2dd908cf2e90))

- Remove settings singleton from config module
  ([`6631adf`](https://github.com/brolnickij/yt-dbl/commit/6631adf66ec9f311b156fb59c817dbb10063bd09))

- Remove unused model_manager singleton
  ([`3ef4512`](https://github.com/brolnickij/yt-dbl/commit/3ef45129bae837832e5fc6beab4b488bed6f7e85))

### Testing

- Add @pytest.mark.parametrize to data-driven tests
  ([`b22efba`](https://github.com/brolnickij/yt-dbl/commit/b22efba1e4502e0873b40f2d69b68b098d1f9051))

- Add Arabic, Hindi, Thai language detection coverage
  ([`0307488`](https://github.com/brolnickij/yt-dbl/commit/0307488bad117df8df8f738f16a053058f110b59))

- Add Claude API error path tests for translate step
  ([`67a4b38`](https://github.com/brolnickij/yt-dbl/commit/67a4b3898deae6a682bbaff891bfc0db931e8c48))

- Add config auto-detection and RAM tier threshold tests
  ([`a1a8f70`](https://github.com/brolnickij/yt-dbl/commit/a1a8f70c13683ba7b25ccaae9fecfbdc0a5442bd))

- Add coverage for utils/memory.py GPU cleanup
  ([`6485a01`](https://github.com/brolnickij/yt-dbl/commit/6485a01169565a7822b2e9dea1094254ef160898))

- Add E2E tests for transcription step (VibeVoice-ASR + ForcedAligner)
  ([`ee7e449`](https://github.com/brolnickij/yt-dbl/commit/ee7e4495d8a4b85a8fcd5251e496e207807fe165))

Real ASR + alignment on 'Me at the zoo' behind --run-slow: - test_transcribe_produces_segments:
  validates segments, words, speakers - test_idempotent_rerun_uses_cache: verifies segments.json
  cache - Update pipeline E2E docstring: transcribe is now real, not stub

- Add early API key validation tests for PipelineRunner
  ([`8a9a267`](https://github.com/brolnickij/yt-dbl/commit/8a9a2671b29736c3a49fce0ca989d428b7d851bd))

- Add postprocess_segment deess fallback coverage
  ([`74a6cbb`](https://github.com/brolnickij/yt-dbl/commit/74a6cbbef3017f0f4f03a6305e4f5ab58bf4a838))

- Add SynthesisError test for empty TTS result
  ([`d589af4`](https://github.com/brolnickij/yt-dbl/commit/d589af4b6941495a77238d034623820594996f28))

- Add tests for logging utilities and suppress_library_noise
  ([`bc46a73`](https://github.com/brolnickij/yt-dbl/commit/bc46a730b5ee83e3c29eea7388b0ced55e79a2f7))

- Add tests for model registry, manager, memory tracking, and config
  ([`a4a114b`](https://github.com/brolnickij/yt-dbl/commit/a4a114bb25110c6dbd8092a0b33a94779ede49fa))

- Add TTS_LANG_MAP validation tests
  ([`e7356ff`](https://github.com/brolnickij/yt-dbl/commit/e7356ff2b7f0ece2912ae793794a7a609f8bbc62))

- Isolate Settings from .env and YT_DBL_ env vars in tests
  ([`c9e52f0`](https://github.com/brolnickij/yt-dbl/commit/c9e52f0de8f5fe26296a137a917c36aaed7c7e4b))

- Make ffmpeg arg assertions order-agnostic
  ([`d749e50`](https://github.com/brolnickij/yt-dbl/commit/d749e50d8f59432402949521f0d8ec125dd279a2))

- Mock model download check in CLI tests and add API key to settings
  ([`4610444`](https://github.com/brolnickij/yt-dbl/commit/4610444a087451ab237b5bc37a02989f9c358b32))

- Mock settings in TestStatusCommand
  ([`fc88bde`](https://github.com/brolnickij/yt-dbl/commit/fc88bdef107e3fd6689016a87100fcc137c390bb))

- Parametrize language detection and format size tests
  ([`3879d38`](https://github.com/brolnickij/yt-dbl/commit/3879d38bd0dd0cf835284cc92d6b04d1d80d1a0d))

- Remove unused mock in TestStatusCommand
  ([`e1dea19`](https://github.com/brolnickij/yt-dbl/commit/e1dea192dce84c5607ba46a189e81d5a4d813150))

- Reorganize into unit/integration/e2e pyramid
  ([`44626c3`](https://github.com/brolnickij/yt-dbl/commit/44626c3af3a98bb0564a20779cc7fe159601b3a7))

- Update version assertion to match semantic versioning format
  ([`eb2e816`](https://github.com/brolnickij/yt-dbl/commit/eb2e816058d7d87c27305bd436773952a7542f5d))

- Update version assertion to use search for regex matching
  ([`4c5d30b`](https://github.com/brolnickij/yt-dbl/commit/4c5d30bc6944b013099012b37d62a623eed18970))

- **integration**: Update runner mocks for source_language and TTS
  ([`1985699`](https://github.com/brolnickij/yt-dbl/commit/1985699b0ee6f57f7dabf8c17b5a3f3bca691d57))

### Breaking Changes

- First stable release of yt-dbl
