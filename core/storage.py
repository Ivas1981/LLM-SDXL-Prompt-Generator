from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any


def load_or_init(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "prompts" in data:
            data = data["prompts"]
        if not isinstance(data, list):
            return []
        return [item for item in data if isinstance(item, dict)]
    except json.JSONDecodeError:
        return []


def save(path: str | Path, data: list[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=p.name + ".",
        suffix=".tmp",
        dir=str(p.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, p)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


_NUMBER_PREFIX_RE = re.compile(r"^(\d+)")


def next_number(data: list[dict]) -> int:
    """Return max(numeric_prefix) + 1, or 1 if data is empty.

    The result is guaranteed not to collide with any existing numeric
    prefix in `data`. Callers may reserve a number by appending a
    placeholder to `data` before re-running, but the bare calculation
    above is collision-free as long as no two items share the same prefix.
    """
    used = {_extract_number(item) for item in data}
    candidate = 1
    if used:
        candidate = max(used) + 1
    while candidate in used:
        candidate += 1
    return candidate


def _extract_number(item: dict) -> int:
    name = str(item.get("name", ""))
    m = _NUMBER_PREFIX_RE.match(name)
    return int(m.group(1)) if m else 0


def names(data: list[dict]) -> list[str]:
    return [str(item.get("name", "")) for item in data]


def prompts(data: list[dict]) -> list[str]:
    return [str(item.get("prompt", "")) for item in data]


def _normalize(s: str) -> str:
    return " ".join(s.split())


def find_duplicate(
    data: list[dict],
    *,
    name: str | None = None,
    prompt: str | None = None,
) -> str | None:
    """Return the field that conflicts, or None.

    Checks exact name match and exact prompt match (whitespace-normalized).
    Empty or whitespace-only strings are treated as not duplicates.
    """
    if name and name.strip():
        for item in data:
            if _normalize(str(item.get("name", ""))) == _normalize(name):
                return "name"
    if prompt and prompt.strip():
        target = _normalize(prompt)
        for item in data:
            if _normalize(str(item.get("prompt", ""))) == target:
                return "prompt"
    return None