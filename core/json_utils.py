from __future__ import annotations

import json
import math
import re
from typing import Any, Iterable


def normalize_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k).strip(): normalize_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [normalize_keys(x) for x in obj]
    return obj


def strip_code_fence(text: str) -> str:
    if not text:
        return ""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json|jsonc|javascript)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    return cleaned


def strip_think_blocks(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_json_object(text: str) -> Any | None:
    if not text:
        return None
    cleaned = strip_think_blocks(strip_code_fence(text))
    if not cleaned:
        return None

    decoder = json.JSONDecoder()
    for start in range(len(cleaned)):
        if cleaned[start] not in "{[":
            continue
        try:
            obj, _ = decoder.raw_decode(cleaned[start:])
            return normalize_keys(obj)
        except json.JSONDecodeError:
            continue

    brace_start = cleaned.find("{")
    brace_end = cleaned.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        candidate = cleaned[brace_start:brace_end + 1]
        for start in range(len(candidate)):
            if candidate[start] != "{":
                continue
            try:
                return normalize_keys(json.loads(candidate[start:]))
            except json.JSONDecodeError:
                continue
    return None


def cosine_similarity(a: Iterable[float], b: Iterable[float]) -> float:
    va = list(a)
    vb = list(b)
    if not va or not vb or len(va) != len(vb):
        return 0.0
    dot = sum(x * y for x, y in zip(va, vb))
    na = math.sqrt(sum(x * x for x in va))
    nb = math.sqrt(sum(y * y for y in vb))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def remove_forbidden_tags(text: str, forbidden: Iterable[str]) -> str:
    if not text:
        return ""
    result = text
    for tag in forbidden:
        pattern = r",?\s*\b" + re.escape(tag) + r"\b\s*,?"
        result = re.sub(pattern, " ", result, flags=re.IGNORECASE)
    result = re.sub(r"\s+", " ", result).strip()
    parts = [p.strip() for p in result.split(",") if p.strip()]
    return ", ".join(parts)