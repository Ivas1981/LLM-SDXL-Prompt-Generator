"""Centralized registry of environment variables used by the application.

All variables in this module are read from os.environ. There is no fallback
chain (file vs env) — env is the single source of truth here.

Only two variables are intentionally NOT in here:
- LM_STUDIO_URL / LM_STUDIO_OPENAI_URL / LM_API_TOKEN: handled by
  config_loader.py because they may also live in config.toml.
- EMBEDDING_MODEL_NAME: kept as an env-only knob because the user picks the
  model interactively from a menu; the registry would just hide it.
- OUTPUT_JSON: handled by core/config.py because the default is a path
  resolved relative to BASE_DIR.

To add a new variable:
1. Add an entry to ENV_SPEC below (name, type, default, description).
2. Reference it via get(name) anywhere in the code.
"""

from __future__ import annotations

import os
from typing import Any, Callable


class EnvSpec:
    __slots__ = ("name", "default", "coerce", "description")

    def __init__(self, name: str, default: Any, coerce: Callable[[str], Any], description: str):
        self.name = name
        self.default = default
        self.coerce = coerce
        self.description = description


def _int(default: int) -> Callable[[str], int]:
    def _coerce(raw: str) -> int:
        try:
            return int(raw)
        except ValueError:
            return default
    return _coerce


def _float(default: float) -> Callable[[str], float]:
    def _coerce(raw: str) -> float:
        try:
            return float(raw)
        except ValueError:
            return default
    return _coerce


def _str(default: str) -> Callable[[str], str]:
    def _coerce(raw: str) -> str:
        return raw if raw != "" else default
    return _coerce


ENV_SPEC: dict[str, EnvSpec] = {
    "LM_STUDIO_URL": EnvSpec(
        "LM_STUDIO_URL", "", _str(""),
        "LM Studio v1 native endpoint URL. Empty = use config.toml or default.",
    ),
    "LM_STUDIO_OPENAI_URL": EnvSpec(
        "LM_STUDIO_OPENAI_URL", "", _str(""),
        "LM Studio OpenAI-compatible base (used for /v1/embeddings).",
    ),
    "LM_API_TOKEN": EnvSpec(
        "LM_API_TOKEN", "", _str(""),
        "Bearer token for LM Studio auth. Empty = interactive prompt on 401.",
    ),
    "UNIQUENESS_THRESHOLD": EnvSpec(
        "UNIQUENESS_THRESHOLD", 0.85, _float(0.85),
        "Cosine similarity threshold above which a new scene is rejected as a duplicate.",
    ),
    "MAX_ATTEMPTS_MULTIPLIER": EnvSpec(
        "MAX_ATTEMPTS_MULTIPLIER", 10, _int(10),
        "max_attempts = target_count * this. Caps total LLM calls per run.",
    ),
    "CHAT_TIMEOUT": EnvSpec(
        "CHAT_TIMEOUT", 600, _int(600),
        "Timeout for LM Studio chat requests in seconds.",
    ),
    "LM_CONTEXT_LENGTH": EnvSpec(
        "LM_CONTEXT_LENGTH", 8192, _int(8192),
        "Context window passed to POST /api/v1/models/load.",
    ),
    "NEGATIVE_BASE_TAGS": EnvSpec(
        "NEGATIVE_BASE_TAGS", "", _str(""),
        "Comma-separated baseline tags always included in the final negative prompt.",
    ),
    "LM_STUDIO_LOG_ROOT": EnvSpec(
        "LM_STUDIO_LOG_ROOT", "", _str(""),
        "Override path to LM Studio server logs. Empty = ~/.lmstudio/server-logs.",
    ),
    "PROMPTGEN_CONFIG": EnvSpec(
        "PROMPTGEN_CONFIG", "", _str(""),
        "Override path to config.toml. Empty = ./config.toml next to main.py.",
    ),
    "CONTEXT_TOKEN": EnvSpec(
        "CONTEXT_TOKEN", "{prompt}", _str("{prompt}"),
        "Placeholder for appearance description in the positive prompt.",
    ),
    "OUTPUT_JSON": EnvSpec(
        "OUTPUT_JSON", "", _str(""),
        "Output JSON file path. Empty = <project>/sdxl_styles.json.",
    ),
    "DEBUG": EnvSpec(
        "DEBUG", "off", _str("off"),
        "Enable debug logging to debug.log. Values: on/off.",
    ),
    "EMBEDDING_MODEL_NAME": EnvSpec(
        "EMBEDDING_MODEL_NAME", "text-embedding-all-minilm-l6-v2", _str("text-embedding-all-minilm-l6-v2"),
        "Embedding model name used for semantic duplicate checks.",
    ),
}


def get(name: str) -> Any:
    spec = ENV_SPEC.get(name)
    if spec is None:
        raise KeyError(f"Unknown env var: {name}")
    raw = os.environ.get(name, spec.default if isinstance(spec.default, str) else "")
    if raw == "" or raw is None:
        return spec.default
    return spec.coerce(raw)


def get_str(name: str) -> str:
    return str(get(name))


def get_int(name: str) -> int:
    return int(get(name))


def get_float(name: str) -> float:
    return float(get(name))


def all_names() -> list[str]:
    return list(ENV_SPEC.keys())


def describe() -> list[dict[str, Any]]:
    return [
        {"name": s.name, "default": s.default, "description": s.description}
        for s in ENV_SPEC.values()
    ]