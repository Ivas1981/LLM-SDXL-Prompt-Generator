# SDXL Prompt Generator

Python tool that drives a local LLM through LM Studio to generate batches of
Stable Diffusion XL prompts. The pipeline runs 8 steps per scene and checks for
uniqueness via embeddings.

## Features

- 8-step LLM pipeline (concept → environment → pose → state → lighting → camera → assembly → naming) plus an optional local-model post-processing pass that refines the assembled prompt
- JSON-only system prompts with strict schemas
- Automatic quality-bait tag stripping
- Semantic duplicate detection via embedding similarity
- Reasoning model support with automatic `reasoning="off"`
- Atomic JSON output with safe Ctrl+C handling
- Architecture-aware sampling presets for Qwen, Gemma, Mistral, DeepSeek, Nemotron, GPT-OSS
- Configurable via environment variables or `config.toml`
- Cross-field and within-field tag deduplication
- Retry reason logging for failed pipeline steps and batch skips
- SFW/NSFW mode toggle via `nsfw` in `config.toml` or `NSFW` env var
- **Location-aware environment validation** (indoor/underground/outdoor weather/lighting constraints)
- **SDXL-compatible token budgets** (~50 words max for positive prompt)
- **Anti-example-copying prompts** for diverse, original outputs

## Project layout

```
core/
  config.py              # env-driven settings
  consistency.py         # environment consistency validation
  embedding_cache.py     # persistent embedding cache
  json_utils.py          # JSON extraction, cosine similarity, tag cleaning
  lm_client.py           # LM Studio HTTP client
  model_info.py          # model metadata, sampling presets
  pipeline.py            # 8-step pipeline orchestration
  storage.py             # atomic JSON read/write, numbering
  validator.py           # prompt assembly and validation
prompts/                 # system prompts (English, JSON-shaped)
tests/                   # unittest suites
main.py                  # CLI entry point
config_loader.py         # TOML + env config resolution
env_registry.py          # centralized env var registry
live_smoke.py            # live LM Studio smoke test
lm_logs.py               # LM Studio log helper
```

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Make sure LM Studio is running with the local server on port 1234.

## Configure

Configuration is read with the following priority (highest first):

1. Environment variable
2. `config.toml` next to `main.py` (only for `lm_studio.*` keys)
3. Built-in default in code

### Environment variables

All variables are listed in `env_registry.py` and can be inspected at runtime:

```python
from env_registry import describe, all_names
for entry in describe():
    print(entry["name"], entry["default"], "-", entry["description"])
```

| Variable | Type | Default | Notes |
|----------|------|---------|-------|
| `LM_STUDIO_URL` | str | (uses config.toml → default `http://localhost:1234/api/v1`) | v1 native endpoint |
| `LM_STUDIO_OPENAI_URL` | str | `http://localhost:1234/v1` | OpenAI-compat for `/v1/embeddings` |
| `LM_API_TOKEN` | str | `""` | Bearer token; falls back to interactive prompt on 401 |
| `EMBEDDING_MODEL_NAME` | str | `text-embedding-all-minilm-l6-v2` | Selected by user at startup |
| `OUTPUT_JSON` | path | `<project>/sdxl_styles.json` | Output file location |
| `CONTEXT_TOKEN` | str | `{prompt}` | Placeholder for appearance description in the positive prompt. Replace with your own description before using in SDXL. |
| `UNIQUENESS_THRESHOLD` | float | `0.85` | Reject duplicates above this cosine similarity |
| `MAX_ATTEMPTS_MULTIPLIER` | int | `10` | `max_attempts = target * this` |
| `CHAT_TIMEOUT` | int | `600` | Seconds |
| `LM_CONTEXT_LENGTH` | int | `8192` | Sent to `/api/v1/models/load` |
| `NEGATIVE_BASE_TAGS` | str | built-in default | Baseline tags for the negative prompt; configurable via env or `config.toml` |
| `NSFW` | bool | `false` | When `true`, step4_state includes an additional `nudity` field in the output |
| `LM_STUDIO_LOG_ROOT` | path | `~/.lmstudio/server-logs` | Used by `lm_logs.py` |
| `PROMPTGEN_CONFIG` | path | `./config.toml` | Override config.toml location |
| `DEBUG` | bool | `off` | Write requests, responses, and errors to `debug.log` |

### `config.toml` example

See `config.toml.example`. Copy it to `config.toml` and edit. The file is
gitignored because it may contain `api_token`.

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

## `{prompt}` placeholder

The positive prompt always begins with `{prompt}` (the `CONTEXT_TOKEN`). This is a placeholder for the user's own subject description — woman, age, hair color, body type, ethnicity, etc. The LLM is instructed to preserve `{prompt}` verbatim in the output. Before using the prompt in SDXL, replace `{prompt}` with your desired description.

Example: if the generated prompt starts with `{prompt}, clockwork engineer in worn leather apron...`, replace it with `young woman with red hair, slim build, european, clockwork engineer in worn leather apron...`.

## NSFW mode

Set `nsfw = true` in `config.toml` under `[generation]` or `NSFW=true` in the
environment to enable NSFW mode. The behavior is:

- **SFW (default):** `prompts/step4_state_sfw.txt` is used instead of
  `prompts/step4_state.txt`. The SFW prompt omits the `nudity` field.
- **NSFW:** `prompts/step4_state.txt` is used (which already includes a
  `nudity` field) and `prompts/step7_assemble_nsfw.txt` replaces the standard
  step7 assembly prompt. The final positive prompt includes the `nudity`
  tags after the camera field.

## Word budget

The hard limits defined in `core/validator.py` are:

- subject: 6, pose: 7, state: 8, environment: 10, relationships: 8,
  lighting: 6, camera: 8, nudity: 6

Summed without `{prompt}`, `relationships`, and `nudity` the maximum is
~53 words; with all fields and `{prompt}` the practical ceiling is ~57
words (still well within SDXL's ~75-token positive-prompt budget).

## Debug logging

Set `DEBUG=on` in the environment or `debug = true` in `config.toml` to append
detailed logs to `debug.log` in the project root. Each entry includes:

- chat requests (model, system prompt, user prompt, payload)
- chat responses
- HTTP request/response details
- pipeline step results and validation errors
- batch start/end, skip reasons, and added scenes

This is useful for diagnosing issues like premature termination or malformed
LLM outputs.

## Test

```bash
python -m unittest discover tests
```

## Notes

- System prompts are in English and request strict JSON, which removes the
  translation step that the previous Russian-based pipeline had to perform.
- `core/validator.py` is the single source of truth for assembling the final
  positive and negative prompts; the LLM never assembles the final strings.
- Quality-bait tags (`masterpiece`, `8k`, `ultra detailed`, `HDR`, ...) are
  stripped from the positive prompt before it is saved.
- Negative prompt starts with a built-in baseline of common SDXL artifacts and per-scene opposites (e.g. `daylight` for night scenes).
- LM Studio auth: the script reads `LM_API_TOKEN` from the environment, and if
  the server responds `401` it asks for a token once per run.
- JSON output is written atomically (`tempfile` + `os.replace`).
- **Prompt word limits**: subject(6), pose(7), state(8), environment(10), relationships(8), lighting(6), camera(8), nudity(6) — total ~53 words max for SDXL compatibility.
- **`{prompt}` placeholder**: The positive prompt always starts with `{prompt}` (configurable via `CONTEXT_TOKEN`). This is a placeholder for the user's own description — woman, age, hair color, body type, ethnicity, etc. Replace it before using the prompt in SDXL.
- **Location-aware validation**: indoor locations reject outdoor weather (rain, snow, etc.); underground locations reject all sunlight; outdoor locations accept any weather valid for time_of_day.
- **Anti-example-copying**: All prompts explicitly instruct the LLM not to copy examples from the instructions, ensuring diverse, original outputs.
- **`{prompt}` is preserved verbatim** in the output JSON. The LLM is instructed to keep it as a placeholder. Users replace it with their own subject description before generating images.