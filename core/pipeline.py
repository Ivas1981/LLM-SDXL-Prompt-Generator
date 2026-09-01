from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .json_utils import extract_json_object, remove_forbidden_tags
from .lm_client import LMClient
from .validator import assemble_positive, assemble_negative, validate_result
from .consistency import validate_environment
from .embedding_cache import EmbeddingCache
from . import storage
from .config import (
    DEFAULT_CONTEXT_TOKEN,
    PIPELINE_STEPS,
    RECENT_NAMES_CONTEXT,
    MAX_ATTEMPTS_MULTIPLIER,
    UNIQUENESS_THRESHOLD,
)

STEP_PARAMS = {
    "step1_concept.txt": (0.9, 350),
    "step2_environment.txt": (0.7, 300),
    "step3_pose.txt": (0.6, 300),
    "step4_state.txt": (0.8, 300),
    "step5_lighting.txt": (0.5, 250),
    "step6_camera.txt": (0.4, 250),
    "step7_assemble.txt": (0.2, 700),
    "step8_name.txt": (0.3, 80),
}

STEP_USER_HINT = {
    "step1_concept.txt": "Generate a new concept. Avoid the following existing concepts:\n{names}",
    "step2_environment.txt": "Concept: {step1_concept}. Generate a detailed environment.",
    "step3_pose.txt": "Concept: {step1_concept}\nEnvironment: {step2_environment}\nGenerate clothing and pose (eye contact with camera required).",
    "step4_state.txt": "Concept: {step1_concept}\nEnvironment: {step2_environment}\nClothing/pose: {step3_pose}\nDescribe a natural way the scene includes partial or full nudity.",
    "step5_lighting.txt": "Environment: {step2_environment}\nDescribe concrete physical lighting for the scene.",
    "step6_camera.txt": "Scene so far: {step1_concept}, {step2_environment}, {step3_pose}, {step4_state}, {step5_lighting}\nDescribe the technical camera parameters.",
    "step7_assemble.txt": "Concept: {step1_concept}\nEnvironment: {step2_environment}\nClothing/pose: {step3_pose}\nState: {step4_state}\nLighting: {step5_lighting}\nCamera: {step6_camera}\nAssemble the final JSON for SDXL.",
    "step8_name.txt": "Concept: {step1_concept}\nEnvironment: {step2_environment}\nPose: {step3_pose}\nAvoid the following existing identifiers:\n{names}\nGenerate a short identifier in English.",
}


def _read_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def load_system_prompts(prompts_dir: Path) -> dict[str, str]:
    prompts = {}
    for name in PIPELINE_STEPS:
        p = prompts_dir / name
        if not p.exists():
            raise FileNotFoundError(f"Missing prompt file: {p}")
        prompts[name] = _read_prompt(p)
    return prompts


def _format_user_hint(step_name: str, ctx: dict[str, str], existing_names: list[str]) -> str:
    template = STEP_USER_HINT.get(step_name, "")
    recent = "\n".join(f"- {n}" for n in existing_names[-RECENT_NAMES_CONTEXT:]) or "none"
    if not template:
        return ""
    if not ctx:
        return template.replace("{names}", recent) if "{names}" in template else ""
    placeholder = "\x00NAMES\x00"
    safe_template = template.replace("{names}", placeholder)
    try:
        formatted = safe_template.format(**ctx)
    except KeyError:
        formatted = safe_template
    return formatted.replace(placeholder, recent)


def _safe_name(raw: str, max_len: int = 60) -> str:
    s = re.sub(r"[^a-z0-9_]", "_", (raw or "").lower())
    s = re.sub(r"_+", "_", s).strip("_")
    if len(s) > max_len:
        s = s[:max_len].rstrip("_")
    return s or "scene"


ENV_STEP = "step2_environment.txt"
ENV_MAX_RETRIES = 3


def _chat_step(lm: LMClient, model: str, system: str, user: str, temp: float, max_tokens: int) -> str | None:
    return lm.chat(model, system, user, temperature=temp, max_tokens=max_tokens)


def _chat_with_retry(
    lm: LMClient,
    model: str,
    prompts: dict[str, str],
    step: str,
    ctx: dict[str, str],
    existing_names: list[str],
    max_attempts: int = 2,
) -> str | None:
    """Call lm.chat for `step` with up to `max_attempts` retries on empty results."""
    temp, max_tokens = STEP_PARAMS.get(step, (0.7, 400))
    for _ in range(max_attempts):
        user_msg = _format_user_hint(step, ctx, existing_names)
        response = _chat_step(lm, model, prompts[step], user_msg, temp, max_tokens)
        if response:
            return response
    return None


def _run_environment_step(
    lm: LMClient,
    model: str,
    prompts: dict[str, str],
    ctx: dict[str, str],
    existing_names: list[str],
) -> dict[str, Any] | None:
    """Run step2 with consistency checks. Up to ENV_MAX_RETRIES attempts."""
    temp, max_tokens = STEP_PARAMS[ENV_STEP]
    base_user = _format_user_hint(ENV_STEP, ctx, existing_names)
    feedback = ""
    for attempt in range(ENV_MAX_RETRIES):
        user_msg = base_user + feedback
        response = _chat_step(lm, model, prompts[ENV_STEP], user_msg, temp, max_tokens)
        if not response:
            feedback = "\n\nYour previous answer was empty. Try again."
            continue
        parsed = extract_json_object(response)
        if not isinstance(parsed, dict):
            feedback = "\n\nYour previous answer was not valid JSON. Output only the JSON object."
            continue
        err = validate_environment(parsed)
        if not err:
            return parsed
        feedback = f"\n\nYour previous environment had a problem: {err}. Fix it and output only the JSON."
    return None


def run_pipeline(
    lm: LMClient,
    model: str,
    prompts: dict[str, str],
    existing_names: list[str],
    context_token: str = DEFAULT_CONTEXT_TOKEN,
    data: list[dict] | None = None,
) -> dict[str, Any] | None:
    ctx: dict[str, str] = {}

    step1 = "step1_concept.txt"
    temp, max_tokens = STEP_PARAMS[step1]
    response = lm.chat(model, prompts[step1], _format_user_hint(step1, ctx, existing_names), temperature=temp, max_tokens=max_tokens)
    if not response:
        return None
    ctx[step1.replace(".txt", "")] = response

    env_result = _run_environment_step(lm, model, prompts, ctx, existing_names)
    if not env_result:
        return None
    ctx[ENV_STEP.replace(".txt", "")] = json.dumps(env_result, ensure_ascii=False)

    remaining_steps = [s for s in PIPELINE_STEPS[:-1] if s not in (step1, ENV_STEP)]
    for step in remaining_steps:
        response = _chat_with_retry(lm, model, prompts, step, ctx, existing_names, max_attempts=2)
        if not response:
            return None
        ctx[step.replace(".txt", "")] = response

    assemble_response = ctx["step7_assemble"]
    parsed = extract_json_object(assemble_response or "")
    if not isinstance(parsed, dict):
        return None

    parts = {
        "subject": parsed.get("subject") or parsed.get("prompt_subject") or "",
        "pose": parsed.get("pose") or "",
        "state": parsed.get("state") or parsed.get("nudity") or "",
        "environment": parsed.get("environment") or parsed.get("setting") or "",
        "relationships": parsed.get("relationships") or parsed.get("spatial") or "",
        "lighting": parsed.get("lighting") or "",
        "camera": parsed.get("camera") or "",
    }

    pos = assemble_positive(parts, context_token=context_token)
    neg = assemble_negative(parts)

    err = validate_result({"prompt": pos, "negative_prompt": neg})
    if err:
        return None

    name_resp = _chat_with_retry(
        lm,
        model,
        prompts,
        "step8_name.txt",
        ctx,
        existing_names,
        max_attempts=2,
    )

    scene_name = _safe_name(name_resp or "")
    if not scene_name or scene_name == "scene":
        subject_words = (parts.get("subject") or "").split(",")[0].strip()
        environment_words = (parts.get("environment") or "").split(",")[0].strip()
        fallback = "_".join(filter(None, [subject_words, environment_words]))
        scene_name = _safe_name(fallback or "scene")

    for _ in range(3):
        conflict = False
        if data:
            for item in data:
                existing_name = str(item.get("name", ""))
                if existing_name == scene_name or existing_name.endswith("_" + scene_name):
                    conflict = True
                    break
        if not conflict:
            break
        name_resp = _chat_with_retry(
            lm,
            model,
            prompts,
            "step8_name.txt",
            ctx,
            existing_names,
            max_attempts=2,
        )
        scene_name = _safe_name(name_resp or "")
        if not scene_name or scene_name == "scene":
            subject_words = (parts.get("subject") or "").split(",")[0].strip()
            environment_words = (parts.get("environment") or "").split(",")[0].strip()
            fallback = "_".join(filter(None, [subject_words, environment_words]))
            scene_name = _safe_name(fallback or "scene")
    else:
        base = scene_name or "scene"
        idx = 1
        while True:
            candidate = f"{base}_{idx}"
            conflict = False
            if data:
                for item in data:
                    existing_name = str(item.get("name", ""))
                    if existing_name == candidate or existing_name.endswith("_" + candidate):
                        conflict = True
                        break
            if not conflict:
                break
            idx += 1
        scene_name = f"{base}_{idx}"

    result = {
        "prompt": pos,
        "negative_prompt": neg,
        "_parts": parts,
        "_raw_assembled": {
            "prompt": parsed.get("prompt", ""),
            "negative_prompt": parsed.get("negative_prompt", ""),
        },
        "_name_raw": name_resp or "",
    }

    result["_scene_name"] = scene_name
    return result


def generate_batch(
    lm: LMClient,
    model: str,
    prompts: dict[str, str],
    data: list[dict],
    target_count: int,
    context_token: str = DEFAULT_CONTEXT_TOKEN,
    on_progress=None,
    save_path=None,
) -> int:
    """Generate `target_count` new entries into `data`.

    `save_path`, if given, is used to atomically save the file after every
    successful addition so a crash never loses more than the in-flight
    attempt. Without a path the caller is responsible for persistence.
    """
    added = 0
    attempts = 0
    max_attempts = max(1, target_count * MAX_ATTEMPTS_MULTIPLIER)
    next_number = _seed_number(data)
    cache = EmbeddingCache()

    while added < target_count and attempts < max_attempts:
        attempts += 1
        existing_names = storage.names(data)
        if on_progress:
            on_progress(added, target_count, attempts)

        result = run_pipeline(lm, model, prompts, existing_names, context_token, data=data)
        if not result:
            continue

        # Reserve a name slot only for entries that pass all checks. The
        # scene_name is built from the current next_number, which is
        # incremented only when the entry is actually appended below.
        scene_name = f"{next_number:02d}_{result['_scene_name']}"

        # Exact-match prompt duplicate check (cheap, no LLM call). Name collision is
        # impossible because next_number is incremented before this point and yields
        # a unique "{NN}_{...}" prefix.
        conflict = storage.find_duplicate(
            data,
            name=None,
            prompt=result["prompt"],
        )
        if conflict:
            print(f"[skip] scene '{scene_name}' has duplicate {conflict}")
            continue

        # Semantic similarity check via embedding cache.
        concept = " ".join(
            str(result["_parts"].get(k, ""))
            for k in ("subject", "pose", "state", "environment")
        )
        existing_concepts = storage.prompts(data)
        score, key = lm.max_similarity_with_cache(concept, existing_concepts, cache)
        if not existing_concepts:
            pass  # First scene, nothing to compare.
        elif score == 0.0 and key is None:
            print(f"[warn] embeddings unavailable for scene '{scene_name}'; "
                  "semantic dup-check skipped")
        elif score > UNIQUENESS_THRESHOLD:
            print(f"[skip] scene '{scene_name}' too similar to existing "
                  f"(score={score:.2f})")
            continue

        data.append({
            "name": scene_name,
            "prompt": result["prompt"],
            "negative_prompt": result["negative_prompt"],
        })
        added += 1
        next_number += 1

        if save_path is not None:
            storage.save(save_path, data)

    return added


def _seed_number(data: list[dict]) -> int:
    from .storage import next_number
    return next_number(data)