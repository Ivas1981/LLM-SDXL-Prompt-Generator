from __future__ import annotations

import json
import re
from typing import Any

from .json_utils import remove_forbidden_tags, extract_json_object
from .config import DEFAULT_CONTEXT_TOKEN, FORBIDDEN_TAGS, NEGATIVE_BASE, QUALITY_BAIT_TAGS, NSFW


MAX_WORDS_PER_FIELD = {
    "subject": 6,
    "pose": 7,
    "state": 8,
    "environment": 10,
    "relationships": 8,
    "lighting": 6,
    "camera": 8,
    "nudity": 6,
}


def _looks_like_prose(text: str) -> bool:
    if not text:
        return False
    stripped = text.strip()
    if stripped.startswith("{prompt}"):
        return False
    return any(c in stripped for c in ".!?")


def _deduplicate_tags(text: str) -> str:
    if not text:
        return ""
    parts = [p.strip() for p in text.split(",") if p.strip()]
    seen: set[str] = set()
    out: list[str] = []
    for part in parts:
        key = part.lower()
        if key not in seen:
            seen.add(key)
            out.append(part)
    return ", ".join(out)


def _clean_field(text: str, max_words: int = 20) -> str:
    if not text:
        return ""
    text = text.strip().strip('"').strip("'")
    text = re.sub(r"\s+", " ", text)
    if "{prompt}" in text:
        prefix, _, suffix = text.partition("{prompt}")
        suffix_clean = _clean_field(suffix, max_words=max_words)
        if suffix_clean:
            return "{prompt}, " + suffix_clean
        return "{prompt}"
    if _looks_like_prose(text):
        words = [w.strip(",.!?;:\"'()[]{} ").lower() for w in re.split(r"[,;]\s*|\s+", text) if w.strip()]
        words = [w for w in words if len(w) > 1][:max_words]
        return _deduplicate_tags(", ".join(words))
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        parts = [p.strip() for p in re.split(r"\s+", text) if p.strip()]
    parts = parts[:max_words]
    return _deduplicate_tags(", ".join(parts))


def clean_step_fields(data: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, str):
            max_words = MAX_WORDS_PER_FIELD.get(key, 20)
            out[key] = _clean_field(value, max_words=max_words)
        else:
            out[key] = value
    return out


def clean_step_json(raw: str) -> str:
    if not raw:
        return raw
    obj = extract_json_object(raw)
    if not isinstance(obj, dict):
        return raw
    cleaned = clean_step_fields(obj)
    return json.dumps(cleaned, ensure_ascii=False)


def assemble_positive(parts: dict[str, str], context_token: str = DEFAULT_CONTEXT_TOKEN) -> str:
    subject = str(parts.get("subject", "")).strip()
    pose = str(parts.get("pose", "")).strip()
    state = str(parts.get("state", "")).strip()
    environment = str(parts.get("environment", "")).strip()
    lighting = str(parts.get("lighting", "")).strip()
    camera = str(parts.get("camera", "")).strip()
    relationships = str(parts.get("relationships", "")).strip()

    if context_token in subject:
        chunks = [subject]
    elif subject:
        chunks = [context_token, subject]
    else:
        chunks = [context_token]

    for part in (pose, state, environment, relationships, lighting, camera):
        if part:
            chunks.append(part)
    if NSFW:
        nudity = str(parts.get("nudity", "")).strip()
        if nudity:
            chunks.append(nudity)
    raw = ", ".join(chunks)
    cleaned = remove_forbidden_tags(raw, FORBIDDEN_TAGS)
    cleaned = remove_quality_bait(cleaned)
    return _deduplicate_tags(cleaned)


def remove_quality_bait(text: str) -> str:
    if not text:
        return ""
    result = text
    for tag in QUALITY_BAIT_TAGS:
        escaped = re.escape(tag)
        if " " in tag or "-" in tag:
            parts = re.split(r"[\s\-]+", tag)
            joined = r"[\s,]+".join(re.escape(p) for p in parts)
            pattern = r"(?:^|[\s,]+)(?:" + joined + r")(?=[\s,]+|$)"
            result = re.sub(pattern, " ", result, flags=re.IGNORECASE)
        else:
            pattern = r",?\s*\b" + escaped + r"\b\s*,?"
            result = re.sub(pattern, " ", result, flags=re.IGNORECASE)
    parts = [p.strip() for p in result.split(",") if p.strip()]
    return ", ".join(parts)


def assemble_negative(context: dict[str, Any], ctx: dict[str, str] | None = None) -> str:
    extras = []
    lighting = str(context.get("lighting", "")).lower()
    environment = str(context.get("environment", "")).lower()

    time_of_day = ""
    weather = ""
    if ctx:
        time_of_day = str(ctx.get("step2_environment", "")).lower()
        try:
            env_obj = json.loads(ctx.get("step2_environment", "{}"))
            if isinstance(env_obj, dict):
                time_of_day = str(env_obj.get("time_of_day", "")).lower()
                weather = str(env_obj.get("weather", "")).lower()
        except (json.JSONDecodeError, TypeError):
            pass

    night_markers = ("night", "dark", "evening", "dusk", "dusky", "moonlight", "candlelight", "twilight")
    if any(m in lighting or m in environment or m in time_of_day for m in night_markers):
        extras.extend(["daylight", "sunlight", "harsh noon light"])

    day_markers = ("morning", "noon", "afternoon", "golden hour", "dawn")
    if any(m in time_of_day for m in day_markers):
        extras.extend(["night", "darkness", "moonlight", "starlight"])

    nature_markers = ("forest", "mountain", "beach", "desert", "jungle", "river", "lake", "ocean")
    if any(m in environment for m in nature_markers):
        extras.extend(["urban", "city street", "skyscraper", "subway"])

    indoor_markers = ("bedroom", "bathroom", "kitchen", "office", "library", "locker", "shower", "studio")
    if any(m in environment for m in indoor_markers):
        extras.extend(["outdoor", "sky", "horizon", "landscape"])

    underground_markers = ("metro", "subway", "cave", "tunnel", "basement", "bunker")
    if any(m in environment for m in underground_markers):
        extras.extend(["sunlight", "open sky", "natural daylight"])

    rain_markers = ("rain", "wet", "dripping", "flooded", "waterlogged")
    if any(m in weather or m in environment for m in rain_markers):
        extras.extend(["dry", "arid", "sunny"])

    pieces = [p for p in (NEGATIVE_BASE, *extras) if p]
    raw = ", ".join(pieces)
    return raw


def validate_result(result: dict[str, Any]) -> str | None:
    if not result:
        return "empty result"
    pos = result.get("prompt", "")
    if not pos or len(pos) < 20:
        return f"positive prompt too short ({len(pos)} chars)"
    if DEFAULT_CONTEXT_TOKEN not in pos:
        return "positive prompt missing context token"
    parts = result.get("_parts") or {}
    for key, value in parts.items():
        if not isinstance(value, str):
            continue
        if _looks_like_prose(value):
            return f"{key} contains prose instead of tags"
        max_words = MAX_WORDS_PER_FIELD.get(key)
        if max_words and len(value.split()) > max_words:
            return f"{key} too long ({len(value.split())} words, max {max_words})"
    if NSFW:
        nudity = parts.get("nudity", "")
        if isinstance(nudity, str) and nudity:
            max_nudity = MAX_WORDS_PER_FIELD.get("nudity", 6)
            if len(nudity.split()) > max_nudity:
                return f"nudity too long ({len(nudity.split())} words, max {max_nudity})"
    return None