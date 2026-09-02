"""Loads runtime configuration from config.toml and env variables.

Resolution priority (highest first):
1. Environment variable (LM_STUDIO_URL, LM_API_TOKEN, ...)
2. Value from config.toml in the project root
3. Built-in default in this module

Env-only overrides are also accepted for path resolution:
- PROMPTGEN_CONFIG: override the location of config.toml.
"""

from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path
from typing import Any

import env_registry


_CONFIG_PATH_ENV = "PROMPTGEN_CONFIG"
DEFAULT_CONFIG_NAME = "config.toml"


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _resolve_config_path() -> Path | None:
    override = env_registry.get_str(_CONFIG_PATH_ENV)
    if override:
        p = Path(override)
        return p if p.exists() else None
    candidate = _project_root() / DEFAULT_CONFIG_NAME
    return candidate if candidate.exists() else None


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except (tomllib.TOMLDecodeError, OSError):
        return {}


def _get(d: dict[str, Any], section: str, key: str) -> Any | None:
    sec = d.get(section)
    if not isinstance(sec, dict):
        return None
    return sec.get(key)


def resolve_url() -> str:
    """Resolve the v1 native endpoint URL."""
    env = env_registry.get_str("LM_STUDIO_URL")
    if env:
        return env
    path = _resolve_config_path()
    if path:
        cfg = _load_toml(path)
        v1 = _get(cfg, "lm_studio", "url")
        if isinstance(v1, str) and v1:
            return v1
        host = _get(cfg, "lm_studio", "host")
        port = _get(cfg, "lm_studio", "port")
        if host and port:
            scheme = "https" if _get(cfg, "lm_studio", "use_ssl") else "http"
            return f"{scheme}://{host}:{port}/api/v1"
    return "http://localhost:1234/api/v1"


def resolve_openai_url() -> str:
    env = env_registry.get_str("LM_STUDIO_OPENAI_URL")
    if env:
        return env
    path = _resolve_config_path()
    if path:
        cfg = _load_toml(path)
        v = _get(cfg, "lm_studio", "openai_url")
        if isinstance(v, str) and v:
            return v
    return "http://localhost:1234/v1"


def resolve_api_token() -> str | None:
    env = env_registry.get_str("LM_API_TOKEN")
    if env:
        return env
    path = _resolve_config_path()
    if path:
        cfg = _load_toml(path)
        v = _get(cfg, "lm_studio", "api_token")
        if isinstance(v, str) and v:
            return v
    return None


def resolve_debug() -> bool:
    raw = os.environ.get("DEBUG")
    if raw is not None:
        return raw.lower() in ("on", "1", "true", "yes")
    path = _resolve_config_path()
    if path:
        cfg = _load_toml(path)
        v = _get(cfg, "lm_studio", "debug")
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower() in ("on", "1", "true", "yes")
    return False


def config_path() -> Path | None:
    """Return the path of the loaded config file, or None if absent."""
    return _resolve_config_path()