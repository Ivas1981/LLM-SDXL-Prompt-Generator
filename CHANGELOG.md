# Changelog

## [1.5.1] - 2026-09-05

### Fixed
- **Test isolation**: `tests/test_config_loader.py` now points `PROMPTGEN_CONFIG` at a temp file and clears all relevant env vars in `setUp`, so a real `config.toml` next to the project never leaks into test runs.
- **`remove_quality_bait`**: multi-word tags like `"ultra detailed"` are now stripped even when the model breaks them across commas (`"ultra, detailed"`).
- **`validate_environment`**: `time_of_day` aliases (`nighttime`, `midnight`, `daytime`, `sunrise`, `sunset`, `twilight`) are now normalized to their canonical values, preventing false validation failures for synonyms.
- **`validate_result`**: NSFW nudity length is now enforced against `MAX_WORDS_PER_FIELD["nudity"] = 6`.
- **Pipeline scene-name collision loop**: replaced the misleading `for/else` with an explicit `resolved` flag for readability.
- **CHANGELOG 1.4.0 lied** about removing `endswith` suffix check — clarified that the check is kept as a safety guard.
- **README NSFW description** corrected: SFW swaps `step4_state.txt` for `step4_state_sfw.txt`; NSFW uses the base `step4_state.txt` (which already contains `nudity`) plus `step7_assemble_nsfw.txt`.

### Changed
- **Mood now flows downstream**: `step1_concept.txt` keeps the `mood` field; `pipeline.run_pipeline` extracts it from step1 JSON and threads it through the user-hint for steps 2, 3, 5, 7.
- **Step3 no longer asks for a separate `eye_contact` field** — eye contact is folded into the `pose` phrase as instructed by the new prompt.
- **`live_smoke.py`**: now calls `run_pipeline()` instead of duplicating its logic, so any pipeline change is reflected automatically.
- **`FORBIDDEN_TAGS_NEGATIVE`** removed (it was an empty tuple never used in production).
- **`MAX_WORDS_PER_FIELD["clothing"]`** removed — clothing is part of `subject` in the final prompt.

## [1.5.0] - 2026-09-04

### Changed
- **`{prompt}` preserved as user placeholder**: The `{prompt}` token in the positive prompt is now kept verbatim in the output. Users replace it themselves with their own description (woman, age, hair color, body type, ethnicity). Previously the pipeline replaced `{prompt}` with the profession from step1, which was incorrect.
- Removed `{prompt}` replacement logic from `_chat_with_retry` and `_format_user_hint` in `core/pipeline.py` — the LLM now sees `{prompt}` in the system prompt and outputs it in the subject field.
- Updated `step7_assemble.txt` and `step7_assemble_nsfw.txt` instructions: `{prompt}` MUST be preserved as a placeholder, not replaced with any text.
- `CONTEXT_TOKEN` (`{prompt}`) is now the definitive placeholder for user-customizable appearance attributes.

### Fixed
- `_format_user_hint` had dead code that replaced `{prompt}` with the profession in user messages — removed.
- `_chat_with_retry` had dead code that replaced `{prompt}` with the profession in system prompts — removed.

## [1.4.0] - 2026-09-02

### Added
- **Location-aware environment validation** in `core/consistency.py`:
  - Indoor locations (kitchen, office, mall, etc.) reject outdoor weather (rain, snow, storm, fog, wind)
  - Underground locations (metro, subway, cave, tunnel, basement) reject ALL sunlight + outdoor weather
  - Indoor but not underground allows window light but rejects harsh direct sunlight
  - Word-boundary regex matching prevents false positives (e.g., "mall" in "small")
- **SDXL-compatible token budgets** across all system prompts:
  - Reduced word limits: subject(6), pose(7), state(8), environment(10), relationships(8), lighting(6), camera(8)
  - Total positive prompt target: ~53 words max (fits SDXL ~75 token limit)
- **Anti-example-copying instructions** in all 11 prompt files
- **Location-specific light source guidance** in step2 and step5 prompts (indoor/underground/outdoor)
- **Cross-field deduplication rules** in step7: no "direct eye contact" in both pose/relationships, no lighting in environment, no pose in subject
- **Post-process hard limit**: positive prompt MUST stay under 75 tokens / ~50 words
- Comprehensive config.toml.example with all configurable keys

### Changed
- All 11 system prompts rewritten: removed concrete examples models were memorizing, tightened word limits, added anti-copying rules
- `pyproject.toml`: `python_requires = ">=3.10"` (was 3.11)
- `config_loader.py`: Added `tomli` fallback for Python 3.10 compatibility
- Model marker fields: `positive`/`negative` → `prompt`/`negative_prompt` for consistency
- Step8 name collision logic: kept `endswith` suffix check (guards against suffix clashes) but also match the full candidate name to catch duplicates that share the suffix exactly.
- Semantic similarity check: now warns but doesn't skip when embeddings unavailable
- Duplicate detection: filters empty prompts from model marker entries

### Fixed
- NSFW step4 template missing `{names}` placeholder for existing concepts
- Test isolation in config_loader tests: proper config.toml cleanup after each test
- Test_latest_log_path_returns_path: rewritten to use temp directories
- Config_loader import chain: tomllib/tomli fallback now works correctly
- test_valid_environment_ok: updated to use valid outdoor location

## [1.3.1] - 2026-09-02

### Changed
- Removed per-step timing output; only total prompt generation time is printed
- Model marker JSON entry is now inserted after the last existing prompt, not at the beginning

## [1.3.0] - 2026-09-02

### Added
- `prompts/step7_post_process.txt` for automatic prompt refinement after step7 assembly
- Automatic model marker insertion as first JSON entry (`name` = model, empty `positive`/`negative`)
- Per-step and total generation timing printed to console
- Pause-before-exit prompt in `main.py`

### Changed
- Step7 assembly is always followed by local-model post-processing (no config flag)
- JSON output prepends model-only entry when file is empty or missing that model marker
- Batch and pipeline timing reported via `[timing]` lines
- Console remains open until user presses a key after generation completes

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