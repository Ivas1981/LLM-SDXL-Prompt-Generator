# Changelog

## [1.2.0] - 2026-09-02

### Added
- SFW/NSFW mode toggle via `nsfw` in `config.toml` or `NSFW` env var
- `step4_state_sfw.txt` prompt for SFW mode (no nudity field)
- Expanded `NEGATIVE_BASE_TAGS` with standard SDXL artifacts (text, watermark, jpeg artifacts, etc.)
- `pyproject.toml` with `python_requires = ">=3.11"` and optional `tomli` for py310
- `requests.compat.json` replaced with stdlib `json` in `core/lm_client.py`

### Changed
- README deduplication fix (removed duplicate feature bullets)
- `config.toml.example` now includes `[generation]` section with `nsfw` and `models_timeout`
- Negative prompt baseline now covers common SDXL artifacts by default
- System prompts overhauled per audit: `time_of_day` in step2 synced with `VALID_TIMES`, step3 `eye_contact` fixed to `"direct"`, step7 now has neutral SFW example plus `step7_assemble_nsfw.txt` for NSFW mode, step4 SFW/NSFW prompts tightened, step5/6/8 rules strengthened

## [1.1.0] - 2026-09-02

### Added
- Cross-field and within-field tag deduplication in `core/validator.py`
- Retry reason logging for pipeline step failures, chat retries, and batch skips
- `NEGATIVE_BASE_TAGS` built-in default to prevent empty `negative_prompt`
- `models_timeout` support in `config.toml` and via `MODELS_TIMEOUT` env var
- `uniqueness_threshold` support in `config.toml` `[generation]` and via `UNIQUENESS_THRESHOLD` env var

### Changed
- Stricter camera/lens prompt rules in `prompts/step6_camera.txt` to prevent tag splitting
- Assembly prompt (`prompts/step7_assemble.txt`) now enforces camera field contains only technical tags
- `README.md` and `README_RU.md` updated with new features

### Fixed
- `_clean_field()` no longer over-splits compound terms in prose mode
- Debug logging no longer overwrites `debug.log` unexpectedly
- Config loading derives OpenAI URL from native LM Studio URL when `openai_url` is missing

## [1.0.0] - 2026-09-01

### Added
- 8-step LLM pipeline for SDXL prompt generation
- JSON-only system prompts with strict schemas per step
- Embedding-based semantic duplicate detection via LM Studio `/v1/embeddings`
- Automatic `reasoning="off"` for reasoning-capable models
- Architecture-aware sampling presets (Qwen, Gemma, Mistral, DeepSeek, Nemotron, GPT-OSS)
- Atomic JSON output with safe Ctrl+C handling
- Environment variable registry (`env_registry.py`) with typed access
- `config.toml` support for LM Studio connection settings
- Comprehensive unit test suite (73 tests)

### Fixed
- Relative imports in `main.py` for direct execution
- `_format_user_hint` bug where `{names}` conflicted with `str.format()`
- Separate `FORBIDDEN_TAGS_NEGATIVE` list to prevent negative prompt tag stripping
- Extended consistency validation regex for `nighttime`, `rainy`, `sunny`, `golden hour`
- Model info architecture key normalization (`qwen3-vl` → `qwen3vl`)
- Step8 name collision handling with retry and numeric suffix fallback
- Progress callback reporting actual added/attempt counts
- LM Studio model unloading using real `instance_id`

### Changed
- Migrated from monolithic script to modular `core/` package
- Positive/negative prompt assembly centralized in `core/validator.py`
- Quality-bait tags stripped before save
