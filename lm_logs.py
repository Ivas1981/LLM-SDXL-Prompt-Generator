"""Helpers for reading LM Studio server logs during local development.

This module is intentionally read-only and not used by the production
pipeline. Import it from a REPL or a one-off script to inspect what the
local server actually saw.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import env_registry

_DEFAULT_LOG_ROOT = Path.home() / ".lmstudio" / "server-logs"


def _resolve_log_root() -> Path:
    override = env_registry.get_str("LM_STUDIO_LOG_ROOT")
    if override:
        return Path(override)
    return _DEFAULT_LOG_ROOT


DEFAULT_LOG_ROOT = _resolve_log_root()

_LINE_RE = re.compile(
    r"^\[(?P<ts>[\d\-: ]+)\]\[(?P<level>[A-Z]+)\](?P<rest>.*)$"
)

_ENDPOINT_RE = re.compile(r"(?P<method>[A-Z]+)\s+to\s+(?P<endpoint>/\S*)")
_TIMING_RE = re.compile(
    r"prompt eval time =\s+(?P<prompt_ms>[\d.]+)\s*ms\s+/\s+(?P<prompt_tokens>\d+)\s+tokens.*?"
    r"eval time =\s+(?P<gen_ms>[\d.]+)\s*ms\s+/\s+(?P<gen_tokens>\d+)\s+tokens.*?"
    r"total time =\s+(?P<total_ms>[\d.]+)\s*ms",
    re.DOTALL,
)


DEFAULT_MAX_LINES = 5000


def latest_log_path(root: Path | None = None, lookback_files: int = 1) -> Path | None:
    """Return the path to the most recently modified log file.

    `lookback_files` is kept for API symmetry but the result is always the
    single newest file. The number of files rotated by LM Studio per day is
    small (usually 1), so additional filtering rarely matters in practice.
    """
    root = root or DEFAULT_LOG_ROOT
    if not root.exists():
        return None
    candidates = [p for p in root.rglob("*.log") if p.is_file()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def read_lines(path: Path, max_lines: int = DEFAULT_MAX_LINES) -> list[str]:
    """Read up to `max_lines` from the end of the log file.

    To avoid loading huge files into memory, `max_lines` defaults to
    DEFAULT_MAX_LINES (5000). Pass `max_lines=0` to disable the limit (not
    recommended for files larger than a few MB).
    """
    if not path.exists():
        return []
    size = path.stat().st_size
    if size == 0:
        return []
    if max_lines and max_lines > 0:
        chunk = min(size, 1024 * 1024)
        with path.open("rb") as f:
            f.seek(max(0, size - chunk))
            data = f.read().decode("utf-8", errors="replace")
        lines = data.splitlines()
        return lines[-max_lines:]
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def filter_by_level(lines: Iterable[str], level: str) -> list[str]:
    lvl = level.upper()
    return [ln for ln in lines if _LINE_RE.match(ln) and _LINE_RE.match(ln).group("level") == lvl]


def find_errors(lines: Iterable[str]) -> list[str]:
    return filter_by_level(lines, "ERROR")


def endpoint_calls(lines: Iterable[str]) -> list[tuple[str, str]]:
    """Return a list of (method, endpoint) tuples seen in the log lines."""
    result = []
    for ln in lines:
        m = _ENDPOINT_RE.search(ln)
        if m:
            result.append((m.group("method"), m.group("endpoint")))
    return result


def endpoint_counts(lines: Iterable[str]) -> Counter[tuple[str, str]]:
    return Counter(endpoint_calls(lines))


def slowest_calls(lines: Iterable[str], top: int = 5) -> list[dict]:
    """Find the slowest timing entries in the log."""
    blocks: list[dict] = []
    buffer: list[str] = []
    for ln in lines:
        if "print_timing" in ln or "slot release" in ln:
            buffer.append(ln)
        else:
            if buffer:
                joined = "\n".join(buffer)
                m = _TIMING_RE.search(joined)
                if m:
                    blocks.append({
                        "prompt_ms": float(m.group("prompt_ms")),
                        "prompt_tokens": int(m.group("prompt_tokens")),
                        "gen_ms": float(m.group("gen_ms")),
                        "gen_tokens": int(m.group("gen_tokens")),
                        "total_ms": float(m.group("total_ms")),
                    })
                buffer = []
    if buffer:
        joined = "\n".join(buffer)
        m = _TIMING_RE.search(joined)
        if m:
            blocks.append({
                "prompt_ms": float(m.group("prompt_ms")),
                "prompt_tokens": int(m.group("prompt_tokens")),
                "gen_ms": float(m.group("gen_ms")),
                "gen_tokens": int(m.group("gen_tokens")),
                "total_ms": float(m.group("total_ms")),
            })
    blocks.sort(key=lambda b: b["total_ms"], reverse=True)
    return blocks[:top]


def summary(path: Path | None = None, max_lines: int = DEFAULT_MAX_LINES) -> dict:
    """Return a compact summary of the most recent log activity.

    `max_lines` caps how much of the log is read (default 5000 lines from
    the tail). Pass 0 to disable the cap (use only for small files).
    """
    target = path or latest_log_path()
    if not target:
        return {"log": None, "errors": 0, "requests": 0, "endpoints": {}, "slowest": []}
    lines = read_lines(target, max_lines=max_lines)
    return {
        "log": str(target),
        "errors": len(find_errors(lines)),
        "requests": len(endpoint_calls(lines)),
        "endpoints": dict(endpoint_counts(lines)),
        "slowest": slowest_calls(lines),
    }