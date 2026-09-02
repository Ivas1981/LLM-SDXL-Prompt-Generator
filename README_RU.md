# Генератор SDXL промптов

Инструмент на Python, который через локальный LLM в LM Studio генерирует
пачки промптов для Stable Diffusion XL. Конвейер выполняет 8 шагов на сцену
и проверяет уникальность через эмбеддинги.

## Возможности

- 8-шаговый LLM-пайплайн (концепт → окружение → поза → состояние → освещение → камера → сборка → название)
- Системные промпты строго в JSON со схемами
- Автоматическая очистка от quality-bait тегов
- Семантическая проверка дубликатов через эмбеддинги
- Поддержка reasoning-моделей с автоматическим `reasoning="off"`
- Атомарная запись JSON, безопасное прерывание через Ctrl+C
- Presets сэмплирования для архитектур Qwen, Gemma, Mistral, DeepSeek, Nemotron, GPT-OSS
- Конфигурация через переменные окружения или `config.toml`
- Дедупликация тегов внутри полей и между полями
- Логирование причин ретраев и пропусков на уровне пайплайна и батча
- Переключатель SFW/NSFW режима через `nsfw` в `config.toml` или env `NSFW`
- **Валидация окружения с учётом типа локации** (indoor/underground/outdoor — погода/освещение)
- **SDXL-совместимые лимиты токенов** (~50 слов макс. для позитивного промпта)
- **Анти-копирование примеров** в промптах для разнообразных, оригинальных результатов

## Структура проекта

```
core/
  config.py              # настройки через env
  consistency.py         # валидация согласованности окружения
  embedding_cache.py     # кэш эмбеддингов
  json_utils.py          # извлечение JSON, косинусное сходство, очистка тегов
  lm_client.py           # HTTP-клиент LM Studio
  model_info.py          # метаданные моделей, пресеты сэмплирования
  pipeline.py            # оркестрация 8-шагового пайплайна
  storage.py             # атомарное чтение/запись JSON, нумерация
  validator.py           # сборка и валидация промптов
prompts/                 # системные промпты (английский, JSON-формат)
tests/                   # unittest-тесты
main.py                  # CLI-точка входа
config_loader.py         # разрешение конфигурации TOML + env
env_registry.py          # централизованный реестр переменных окружения
live_smoke.py            # быстрый smoke-тест LM Studio
lm_logs.py               # вспомогательная утилита для логов LM Studio
```

## Установка

```bash
pip install -r requirements.txt
```

## Запуск

```bash
python main.py
```

Убедитесь, что LM Studio запущен с локальным сервером на порту 1234.

## Конфигурация

Приоритет чтения конфигурации (от высшего к низшему):

1. Переменная окружения
2. `config.toml` рядом с `main.py` (только ключи `lm_studio.*`)
3. Встроенный дефолт в коде

### Переменные окружения

Все переменные перечислены в `env_registry.py` и могут быть inspected во время выполнения:

```python
from env_registry import describe, all_names
for entry in describe():
    print(entry["name"], entry["default"], "-", entry["description"])
```

| Переменная | Тип | Дефолт | Примечания |
|------------|-----|--------|------------|
| `LM_STUDIO_URL` | str | (используется config.toml → дефолт `http://localhost:1234/api/v1`) | v1 нативный эндпоинт |
| `LM_STUDIO_OPENAI_URL` | str | `http://localhost:1234/v1` | OpenAI-совместимый для `/v1/embeddings` |
| `LM_API_TOKEN` | str | `""` | Bearer токен; при 401 запрашивается интерактивно |
| `EMBEDDING_MODEL_NAME` | str | `text-embedding-all-minilm-l6-v2` | Выбирается пользователем при старте |
| `OUTPUT_JSON` | path | `<project>/sdxl_styles.json` | Путь к выходному файлу |
| `CONTEXT_TOKEN` | str | `{prompt}` | Плейсхолдер для описания внешности |
| `UNIQUENESS_THRESHOLD` | float | `0.85` | Отклонять дубликаты выше этого косинусного сходства |
| `MAX_ATTEMPTS_MULTIPLIER` | int | `10` | `max_attempts = target * это` |
| `CHAT_TIMEOUT` | int | `600` | Секунды |
| `LM_CONTEXT_LENGTH` | int | `8192` | Отправляется в `/api/v1/models/load` |
| `NEGATIVE_BASE_TAGS` | str | встроенный дефолт | Базовые теги для негативного промпта; переопределяется через env или `config.toml` |
| `NSFW` | bool | `false` | При `true` шаг 4 включает дополнительное поле `nudity` в вывод |
| `LM_STUDIO_LOG_ROOT` | path | `~/.lmstudio/server-logs` | Используется `lm_logs.py` |
| `PROMPTGEN_CONFIG` | path | `./config.toml` | Переопределить расположение config.toml |
| `DEBUG` | bool | `off` | Записывать запросы, ответы и ошибки в `debug.log` |

### Пример `config.toml`

См. `config.toml.example`. Скопируйте в `config.toml` и отредактируйте. Файл
в `.gitignore`, так как может содержать `api_token`.

```toml
[lm_studio]
url = "http://localhost:1234/api/v1"
openai_url = "http://localhost:1234/v1"
api_token = ""
debug = false
models_timeout = 180
chat_timeout = 600

[generation]
uniqueness_threshold = 0.85
nsfw = false
lm_context_length = 8192
max_attempts_multiplier = 10
negative_base_tags = "deformed, bad anatomy, ..."
output_json = "sdxl_styles.json"
context_token = "{prompt}"
lm_studio_log_root = ""
promptgen_config = ""
```

## NSFW режим

Установите `nsfw = true` в `config.toml` в секции `[generation]` или
`NSFW=true` в окружении, чтобы включить NSFW режим. В этом режиме шаг 4
(`state`) включает дополнительное поле `nudity` в вывод. По умолчанию
генератор работает в SFW режиме, и шаг `state` выводит только теги физического
состояния.

## Отладка

Установите `DEBUG=on` в окружении или `debug = true` в `config.toml`, чтобы
дописывать детальные логи в `debug.log` в корне проекта. В лог попадают:

- chat-запросы (модель, system, user, payload)
- chat-ответы
- HTTP-запросы/ответы
- результаты шагов пайплайна и ошибки валидации
- старт/окончание батча, причины пропусков, добавленные сцены

Это удобно для диагностики преждевременной остановки или некорректных
ответов модели.

## Тесты

```bash
python -m unittest discover tests
```

## Примечания

- Системные промпты на английском и требуют строгий JSON, что убирает
  шаг перевода, который был в предыдущем русскоязычном пайплайне.
- `core/validator.py` — единственный источник истины для сборки финальных
  позитивного и негативного промптов; LLM никогда не собирает финальные строки.
- Quality-bait теги (`masterpiece`, `8k`, `ultra detailed`, `HDR`...) удаляются
  из позитивного промпта перед сохранением.
- Негативный промпт начинается с встроенного базового набора распространённых SDXL-артефактов; контекст сцены добавляет только релевантные противоположности (например, `daylight` для ночных сцен).
- Авторизация LM Studio: скрипт читает `LM_API_TOKEN` из окружения, а если
  сервер отвечает `401` — запрашивает токен один раз за запуск.
- JSON-выход записывается атомарно (`tempfile` + `os.replace`).
- **Лимиты слов промптов**: subject(6), pose(7), state(8), environment(10), relationships(8), lighting(6), camera(8) — всего ~53 слова макс. для совместимости с SDXL.
- **Валидация с учётом локации**: indoor локации отвергают уличную погоду (дождь, снег и т.д.); underground локации отвергают весь солнечный свет; outdoor локации принимают любую погоду, валидную для time_of_day.
- **Анти-копирование примеров**: Все промпты явно запрещают копировать примеры из инструкций, обеспечивая разнообразные, оригинальные результаты.