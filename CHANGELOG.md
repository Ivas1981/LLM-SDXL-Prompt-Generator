# Changelog

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
