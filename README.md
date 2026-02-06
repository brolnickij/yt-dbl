# yt-dbl
> [!WARNING]
> Работает **только на Apple Silicon** (M1/M2/M3/M4), весь ML-inference выполняется через MLX на Metal GPU
>
> Тестировалось на **M4 Pro** (20-core GPU, 48 GB unified memory)

CLI-инструмент для автоматического дубляжа YouTube-видео с клонированием голоса

Весь ML-inference (ASR, alignment, TTS) выполняется локально на Apple Silicon через [MLX](https://github.com/ml-explore/mlx)

Для перевода используется Claude API

Результат — видеофайл с озвучкой голосом оригинального спикера на целевом языке


## Как это работает
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                YouTube URL                                      │
└─────────────────────────────────────┬───────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  1. DOWNLOAD                                                                    │
│                                                                                 │
│  yt-dlp скачивает видео, ffmpeg извлекает аудиодорожку                          │
│  Выход: video.mp4, audio.wav (48 kHz, mono)                                     │
└─────────────────────────────────────┬───────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  2. SEPARATE                                                                    │
│                                                                                 │
│  BS-RoFormer разделяет аудио на голос и фон (ONNX + CoreML)                     │
│  Выход: vocals.wav, background.wav                                              │
└──────────────────┬──────────────────────────────────────────┬───────────────────┘
                   │                                          │
              vocals.wav                                background.wav
                   │                                          │
                   ▼                                          │
┌──────────────────────────────────────────────────────┐      │
│  3. TRANSCRIBE                                       │      │
│                                                      │      │
│  VibeVoice-ASR (MLX, ~8 GB)                          │      │
│    → сегменты речи + speaker diarization             │      │
│  Qwen3-ForcedAligner (MLX, ~600 MB)                  │      │
│    → word-level timestamps                           │      │
│  + автодетект языка по Unicode-скриптам              │      │
│                                                      │      │
│  Выход: segments.json                                │      │
└──────────────────────────┬───────────────────────────┘      │
                           │                                  │
                           ▼                                  │
┌──────────────────────────────────────────────────────┐      │
│  4. TRANSLATE                                        │      │
│                                                      │      │
│  Claude API (single-pass, все сегменты разом)        │      │
│  Адаптация под TTS: короткие фразы, числа прописью,  │      │
│  без спецсимволов                                    │      │
│                                                      │      │
│  Выход: translations.json, subtitles.srt             │      │
└──────────────────────────┬───────────────────────────┘      │
                           │                                  │
                           ▼                                  │
┌──────────────────────────────────────────────────────┐      │
│  5. SYNTHESIZE                                       │      │
│                                                      │      │
│  Qwen3-TTS (MLX, ~1.7 GB) — voice cloning            │      │
│  по голосовому референсу каждого спикера             │      │
│  Postprocessing (параллельно, ThreadPool):           │      │
│    • speed-up (rubberband или atempo)                │      │
│    • loudnorm (-16 LUFS, 2-pass)                     │      │
│    • de-essing                                       │      │
│                                                      │      │
│  Выход: segment_0000.wav, segment_0001.wav ...       │      │
└──────────────────────────┬───────────────────────────┘      │
                           │                                  │
                           ▼                                  ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  6. ASSEMBLE                                                                    │
│                                                                                 │
│  Речевой трек (crossfade 50 ms, equal-power) + background (sidechain ducking)   │
│  + video (copy) + субтитры (softsub / hardsub / none)                           │
│  Все за один вызов ffmpeg                                                       │
│                                                                                 │
│  Выход: result.mp4                                                              │
└─────────────────────────────────────┬───────────────────────────────────────────┘
                                      │
                                      ▼
                            ┌───────────────────┐
                            │    result.mp4     │
                            └───────────────────┘
```

### Управление памятью
ML-модели загружаются и выгружаются через LRU-менеджер

Количество одновременно загруженных моделей определяется автоматически по объему RAM:

```
RAM              Моделей    Batch (separation)
─────────────    ───────    ──────────────────
≤ 16 GB          1          1
17–31 GB         2          2
32–47 GB         3          4
48+ GB           3          8
```

ASR-модель (~8 GB) выгружается перед загрузкой Aligner, чтобы не держать оба в памяти

### Структура рабочей директории
```
work/
└── <video_id>/
    ├── state.json                  ← чекпоинт пайплайна (JSON)
    ├── 01_download/
    │   ├── video.mp4               ← исходное видео
    │   └── audio.wav               ← извлеченный аудиотрек (48 kHz, mono)
    ├── 02_separate/
    │   ├── vocals.wav              ← изолированный голос
    │   └── background.wav          ← фоновая музыка/шум
    ├── 03_transcribe/
    │   └── segments.json           ← сегменты, спикеры, слова с таймкодами
    ├── 04_translate/
    │   ├── translations.json       ← переведенные тексты
    │   └── subtitles.srt           ← субтитры (SRT)
    ├── 05_synthesize/
    │   ├── ref_SPEAKER_00.wav      ← голосовой референс спикера
    │   ├── segment_0000.wav        ← финальные сегменты (после postprocessing)
    │   ├── segment_0001.wav
    │   └── synth_meta.json         ← метаданные синтеза
    └── 06_assemble/
        └── speech.wav              ← собранный речевой трек
    └── result.mp4                  ← итоговый файл (в корне job dir)
```

## Требования
- **macOS** с **Apple Silicon** (M1/M2/M3/M4) — MLX работает только на Metal
- **Python** ≥ 3.11
- **FFmpeg** — автодетект; предпочитает `ffmpeg-full` (Homebrew) для поддержки rubberband
- **yt-dlp** — скачивание видео
- **Anthropic API key** — для перевода через Claude

## Установка
```bash
git clone <repo-url> && cd yt-dbl
uv sync

# API-ключ (обязателен для шага перевода)
export YT_DBL_ANTHROPIC_API_KEY="sk-ant-..."

# Опционально: ffmpeg-full для pitch-preserving speed-up (rubberband)
brew install ffmpeg-full
```

### Предварительная загрузка моделей
Модели скачиваются автоматически при первом запуске, но можно загрузить заранее (~10.5 GB):

```bash
uv run yt-dbl models download
```

## Быстрый старт
```bash
# Дублировать видео на русский (по умолчанию)
uv run yt-dbl dub "https://www.youtube.com/watch?v=VIDEO_ID"

# Указать целевой язык
uv run yt-dbl dub "https://youtu.be/VIDEO_ID" -t es

# Начать с определенного шага (предыдущие пропускаются)
uv run yt-dbl dub "https://youtu.be/VIDEO_ID" --from-step translate

# Проверить статус задачи
uv run yt-dbl status VIDEO_ID

# Продолжить прерванный дубляж
uv run yt-dbl resume VIDEO_ID
```

## Команды
### `dub` — дублирование видео

```bash
uv run yt-dbl dub <URL> [опции]
```

| Опция | Описание | По умолчанию |
|---|---|---|
| `-t`, `--target-language` | Целевой язык перевода | `ru` |
| `--bg-volume` | Громкость фона (0.0–1.0) | `0.15` |
| `--max-speed` | Макс. ускорение TTS (1.0–2.0) | `1.4` |
| `--max-models` | Макс. моделей в памяти | авто (по RAM) |
| `--from-step` | Начать с шага: `download` / `separate` / `transcribe` / `translate` / `synthesize` / `assemble` | — |
| `--no-subs` | Без субтитров | `false` |
| `--sub-mode` | Режим субтитров: `softsub` / `hardsub` / `none` | `softsub` |
| `--format` | Формат выхода: `mp4` / `mkv` | `mp4` |

### `resume` — продолжить прерванный дубляж
```bash
uv run yt-dbl resume <video_id> [--max-models N]
```

Пайплайн сохраняет `state.json` после каждого шага. При прерывании `resume` продолжит с последнего незавершенного шага

### `status` — статус задачи
```bash
uv run yt-dbl status <video_id>
```

Выводит таблицу с состоянием каждого шага (`pending` / `running` / `completed` / `failed`), временем выполнения и метаданными видео

### `models list` — список ML-моделей
```bash
uv run yt-dbl models list
```

Показывает все модели, их статус загрузки и размер на диске

### `models download` — предварительное скачивание моделей
```bash
uv run yt-dbl models download
```

Загружает все HuggingFace-модели, а модель `audio-separator` скачивается автоматически при первом использовании

## Конфигурация
Настройки загружаются в порядке приоритета:

1. Аргументы CLI
2. Переменные окружения (префикс `YT_DBL_`)
3. Файл `.env`
4. Значения по умолчанию

Пример `.env`:

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

### Все параметры
| Параметр | Env-переменная | По умолчанию | Описание |
|---|---|---|---|
| `target_language` | `YT_DBL_TARGET_LANGUAGE` | `ru` | Целевой язык |
| `output_format` | `YT_DBL_OUTPUT_FORMAT` | `mp4` | `mp4` / `mkv` |
| `subtitle_mode` | `YT_DBL_SUBTITLE_MODE` | `softsub` | `softsub` / `hardsub` / `none` |
| `background_volume` | `YT_DBL_BACKGROUND_VOLUME` | `0.15` | Громкость фона (0.0–1.0) |
| `background_ducking` | `YT_DBL_BACKGROUND_DUCKING` | `true` | Приглушать фон во время речи (sidechain) |
| `max_speed_factor` | `YT_DBL_MAX_SPEED_FACTOR` | `1.4` | Макс. ускорение TTS (1.0–2.0) |
| `voice_ref_duration` | `YT_DBL_VOICE_REF_DURATION` | `7.0` | Длительность голосового референса (3–30 сек) |
| `max_loaded_models` | `YT_DBL_MAX_LOADED_MODELS` | `0` (авто) | Макс. моделей в памяти |
| `anthropic_api_key` | `YT_DBL_ANTHROPIC_API_KEY` | — | Ключ Anthropic API |
| `work_dir` | `YT_DBL_WORK_DIR` | `work` | Рабочая директория |

## Используемые модели
| Модель | Размер | Задача | Inference |
|---|---|---|---|
| [VibeVoice-ASR](https://huggingface.co/mlx-community/VibeVoice-ASR-bf16) | ~8.2 GB | ASR + speaker diarization | MLX (Metal) |
| [Qwen3-ForcedAligner](https://huggingface.co/mlx-community/Qwen3-ForcedAligner-0.6B-8bit) | ~600 MB | Word-level alignment | MLX (Metal) |
| [Qwen3-TTS](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16) | ~1.7 GB | TTS с клонированием голоса | MLX (Metal) |
| MelBand-RoFormer (BS-RoFormer) | ~200 MB | Разделение голоса и фона | ONNX + CoreML |
| Claude Sonnet 4.5 | — | Перевод текста | API (Anthropic) |

### Поддерживаемые языки
**TTS (синтез):** русский, английский, немецкий, французский, испанский, итальянский, португальский, китайский, японский, корейский, арабский, хинди, турецкий, нидерландский, польский, украинский

**ASR (распознавание):** автодетект по Unicode-скриптам (латиница, кириллица, арабица, деванагари, CJK и др.)


## Разработка
```bash
just check          # lint + format + typecheck + tests
just test           # быстрые тесты (parallel, coverage)
just test-e2e       # E2E тесты (нужен FFmpeg + сеть)
just fix            # auto-fix линтера
just format         # auto-format
```

## Лицензия
MIT
