from __future__ import annotations

import re
from typing import Any

from .json_utils import remove_forbidden_tags
from .config import DEFAULT_CONTEXT_TOKEN, FORBIDDEN_TAGS, FORBIDDEN_TAGS_NEGATIVE, NEGATIVE_BASE, QUALITY_BAIT_TAGS


def assemble_positive(parts: dict[str, str], context_token: str = DEFAULT_CONTEXT_TOKEN) -> str:
    subject = parts.get("subject", "").strip()
    pose = parts.get("pose", "").strip()
    state = parts.get("state", "").strip()
    environment = parts.get("environment", "").strip()
    lighting = parts.get("lighting", "").strip()
    camera = parts.get("camera", "").strip()
    relationships = parts.get("relationships", "").strip()

    if context_token in subject:
        chunks = [subject]
    elif subject:
        chunks = [context_token, subject]
    else:
        chunks = [context_token]

    for part in (pose, state, environment, relationships, lighting, camera):
        if part:
            chunks.append(part)
    raw = ", ".join(chunks)
    cleaned = remove_forbidden_tags(raw, FORBIDDEN_TAGS)
    return remove_quality_bait(cleaned)


def remove_quality_bait(text: str) -> str:
    if not text:
        return ""
    result = text
    for tag in QUALITY_BAIT_TAGS:
        pattern = r",?\s*\b" + re.escape(tag) + r"\b\s*,?"
        result = re.sub(pattern, " ", result, flags=re.IGNORECASE)
    parts = [p.strip() for p in result.split(",") if p.strip()]
    return ", ".join(parts)


def assemble_negative(context: dict[str, Any]) -> str:
    extras = []
    lighting = str(context.get("lighting", "")).lower()
    environment = str(context.get("environment", "")).lower()

    night_markers = ("night", "dark", "evening", "dusk", "dusky", "moonlight", "candlelight", "twilight")
    if any(m in lighting or m in environment for m in night_markers):
        extras.extend(["daylight", "sunlight", "harsh noon light"])

    nature_markers = ("forest", "mountain", "beach", "desert", "jungle", "river", "lake", "ocean")
    if any(m in environment for m in nature_markers):
        extras.extend(["urban", "city street", "skyscraper", "subway"])

    indoor_markers = ("bedroom", "bathroom", "kitchen", "office", "library", "locker", "shower", "studio")
    if any(m in environment for m in indoor_markers):
        extras.extend(["outdoor", "sky", "horizon", "landscape"])

    pieces = [p for p in (NEGATIVE_BASE, *extras) if p]
    raw = ", ".join(pieces)
    return remove_forbidden_tags(raw, FORBIDDEN_TAGS_NEGATIVE)


def validate_result(result: dict[str, Any]) -> str | None:
    if not result:
        return "empty result"
    pos = result.get("prompt", "")
    if not pos or len(pos) < 20:
        return f"positive prompt too short ({len(pos)} chars)"
    if DEFAULT_CONTEXT_TOKEN not in pos:
        return "positive prompt missing context token"
    return None