from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .config import ARCH_PRESETS, DEFAULT_CONTEXT_LENGTH


@dataclass
class ModelInfo:
    key: str
    type: str
    architecture: str | None
    quantization: str | None
    bits_per_weight: int | None
    max_context_length: int
    params_string: str | None
    size_bytes: int | None
    vision: bool
    tool_use: bool
    reasoning_allowed: list[str] = field(default_factory=list)
    reasoning_default: str | None = None
    loaded_context_length: int | None = None

    @property
    def is_chat(self) -> bool:
        return self.type in {"llm"}

    @property
    def is_reasoning_capable(self) -> bool:
        return len(self.reasoning_allowed) > 0

    @property
    def display_short(self) -> str:
        parts = [self.key]
        if self.params_string:
            parts.append(self.params_string)
        if self.quantization:
            parts.append(self.quantization)
        return " · ".join(parts)

    def sampling_preset(self) -> dict[str, float]:
        arch = (self.architecture or "").lower()
        arch = re.sub(r"[-_ ]", "", arch)
        return dict(ARCH_PRESETS.get(arch, ARCH_PRESETS["_default"]))

    def suggested_context_length(self, requested: int = DEFAULT_CONTEXT_LENGTH) -> int:
        if self.max_context_length and requested > self.max_context_length:
            return self.max_context_length
        return requested


def parse_v1_model(m: dict[str, Any]) -> ModelInfo:
    caps = m.get("capabilities") or {}
    reasoning = caps.get("reasoning") or {}
    quant = m.get("quantization") or {}
    loaded_ctx = None
    instances = m.get("loaded_instances") or []
    if instances:
        cfg = instances[0].get("config") or {}
        loaded_ctx = cfg.get("context_length")
    return ModelInfo(
        key=str(m.get("key", m.get("id", "unknown"))),
        type=str(m.get("type", "llm")),
        architecture=m.get("architecture"),
        quantization=quant.get("name") if isinstance(quant, dict) else None,
        bits_per_weight=quant.get("bits_per_weight") if isinstance(quant, dict) else None,
        max_context_length=int(m.get("max_context_length", 0) or 0),
        params_string=m.get("params_string"),
        size_bytes=m.get("size_bytes"),
        vision=bool(caps.get("vision", False)),
        tool_use=bool(caps.get("trained_for_tool_use", False)),
        reasoning_allowed=list(reasoning.get("allowed_options", []) or []),
        reasoning_default=reasoning.get("default"),
        loaded_context_length=loaded_ctx,
    )