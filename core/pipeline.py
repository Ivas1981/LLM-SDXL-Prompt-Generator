from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from .json_utils import extract_json_object, remove_forbidden_tags
from .lm_client import LMClient
from .validator import assemble_positive, assemble_negative, validate_result, clean_step_json, clean_step_fields
from .consistency import validate_environment
from .embedding_cache import EmbeddingCache
from . import storage
from .config import (
    DEFAULT_CONTEXT_TOKEN,
    PIPELINE_STEPS,
    RECENT_NAMES_CONTEXT,
    MAX_ATTEMPTS_MULTIPLIER,
    UNIQUENESS_THRESHOLD,
    DEBUG,
    NSFW,
)
from .debug_log import get as get_debug

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

    if not NSFW:
        sfw_state = prompts_dir / "step4_state_sfw.txt"
        if sfw_state.exists():
            prompts["step4_state.txt"] = _read_prompt(sfw_state)
    else:
        nsfw_assemble = prompts_dir / "step7_assemble_nsfw.txt"
        if nsfw_assemble.exists():
            prompts["step7_assemble.txt"] = _read_prompt(nsfw_assemble)

    post_process = prompts_dir / "step7_post_process.txt"
    if post_process.exists():
        prompts["step7_post_process.txt"] = _read_prompt(post_process)

    return prompts


def _format_user_hint(step_name: str, ctx: dict[str, str], existing_names: list[str]) -> str:
    template = STEP_USER_HINT.get(step_name, "")
    if step_name == "step4_state.txt":
        template = (
            "Concept: {step1_concept}\nEnvironment: {step2_environment}\nClothing/pose: {step3_pose}\nDescribe a natural way the scene includes partial or full nudity."
            if NSFW
            else "Concept: {step1_concept}\nEnvironment: {step2_environment}\nClothing/pose: {step3_pose}\nDescribe the physical state and natural expression."
        )
    recent = "\n".join(f"- {n}" for n in existing_names[-RECENT_NAMES_CONTEXT:]) or "none"
    if not template:
        return ""
    result = template.replace("{names}", recent)
    for key, value in ctx.items():
        result = result.replace("{" + key + "}", value)
    return result


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


def _build_parts(cleaned_parsed: dict[str, Any]) -> dict[str, str]:
    return {
        "subject": cleaned_parsed.get("subject") or cleaned_parsed.get("prompt_subject") or "",
        "pose": cleaned_parsed.get("pose") or "",
        "state": cleaned_parsed.get("state") or (cleaned_parsed.get("nudity") if NSFW else "") or "",
        "environment": cleaned_parsed.get("environment") or cleaned_parsed.get("setting") or "",
        "relationships": cleaned_parsed.get("relationships") or cleaned_parsed.get("spatial") or "",
        "lighting": cleaned_parsed.get("lighting") or "",
        "camera": cleaned_parsed.get("camera") or "",
    }


def _post_process_with_local_model(lm: LMClient, model: str, system: str, positive: str, negative: str) -> tuple[str, str]:
    user = f"Positive prompt:\n{positive}\n\nNegative prompt:\n{negative}\n\nRefine both prompts and return JSON."
    response = lm.chat(model, system, user, temperature=0.2, max_tokens=700)
    if not response:
        return positive, negative
    parsed = extract_json_object(response)
    if not isinstance(parsed, dict):
        return positive, negative
    new_pos = parsed.get("positive") or positive
    new_neg = parsed.get("negative") or negative
    if DEFAULT_CONTEXT_TOKEN not in new_pos:
        if new_pos:
            new_pos = f"{DEFAULT_CONTEXT_TOKEN}, {new_pos}"
        else:
            new_pos = DEFAULT_CONTEXT_TOKEN
    return new_pos, new_neg


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
    debug = get_debug() if DEBUG else None
    for attempt in range(max_attempts):
        user_msg = _format_user_hint(step, ctx, existing_names)
        response = _chat_step(lm, model, prompts[step], user_msg, temp, max_tokens)
        if response:
            return response
        reason = "empty_response"
        if DEBUG and debug:
            debug.log("CHAT_RETRY", f"step={step}\nattempt={attempt + 1}\nreason={reason}")
        print(f"  -> retry {step} attempt {attempt + 1}: {reason}")
    return None


def _run_environment_step(
    lm: LMClient,
    model: str,
    prompts: dict[str, str],
    ctx: dict[str, str],
    existing_names: list[str],
) -> tuple[dict[str, Any] | None, str | None]:
    """Run step2 with consistency checks. Up to ENV_MAX_RETRIES attempts."""
    temp, max_tokens = STEP_PARAMS[ENV_STEP]
    base_user = _format_user_hint(ENV_STEP, ctx, existing_names)
    debug = get_debug() if DEBUG else None
    feedback = ""
    for attempt in range(ENV_MAX_RETRIES):
        user_msg = base_user + feedback
        response = _chat_step(lm, model, prompts[ENV_STEP], user_msg, temp, max_tokens)
        if not response:
            reason = "empty_response"
            if DEBUG and debug:
                debug.log("ENV_STEP_RETRY", f"attempt={attempt + 1}\nreason={reason}")
            print(f"  -> retry {ENV_STEP} attempt {attempt + 1}: {reason}")
            feedback = "\n\nYour previous answer was empty. Try again."
            continue
        parsed = extract_json_object(response)
        if not isinstance(parsed, dict):
            reason = "invalid_json"
            if DEBUG and debug:
                debug.log("ENV_STEP_RETRY", f"attempt={attempt + 1}\nreason={reason}")
            print(f"  -> retry {ENV_STEP} attempt {attempt + 1}: {reason}")
            feedback = "\n\nYour previous answer was not valid JSON. Output only the JSON object."
            continue
        err = validate_environment(parsed)
        if not err:
            return parsed, None
        reason = f"validation_failed: {err}"
        if DEBUG and debug:
            debug.log("ENV_STEP_RETRY", f"attempt={attempt + 1}\nreason={reason}")
        print(f"  -> retry {ENV_STEP} attempt {attempt + 1}: {reason}")
        feedback = f"\n\nYour previous environment had a problem: {err}. Fix it and output only the JSON."
    if DEBUG and debug:
        debug.log("ENV_STEP_FAILED", f"reason=max_retries_exceeded\nmax_retries={ENV_MAX_RETRIES}")
    return None, "environment step failed"


def run_pipeline(
    lm: LMClient,
    model: str,
    prompts: dict[str, str],
    existing_names: list[str],
    context_token: str = DEFAULT_CONTEXT_TOKEN,
    data: list[dict] | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    ctx: dict[str, str] = {}
    debug = get_debug() if DEBUG else None
    t0 = time.perf_counter()
    try:
        step1 = "step1_concept.txt"
        temp, max_tokens = STEP_PARAMS[step1]
        user = _format_user_hint(step1, ctx, existing_names)
        if DEBUG and debug:
            debug.log("PIPELINE_STEP", f"step=step1_concept\nuser={user}")
        response = lm.chat(model, prompts[step1], user, temperature=temp, max_tokens=max_tokens)
        if DEBUG and debug:
            debug.log("PIPELINE_STEP_RESULT", f"step=step1_concept\nresponse={response}")
        if not response:
            return None, "empty response (step1)"
        ctx[step1.replace(".txt", "")] = clean_step_json(response)

        env_result, env_reason = _run_environment_step(lm, model, prompts, ctx, existing_names)
        if DEBUG and debug:
            debug.log("PIPELINE_STEP_RESULT", f"step=step2_environment\nresult={env_result}")
        if not env_result:
            return None, env_reason or "environment step failed"
        ctx[ENV_STEP.replace(".txt", "")] = json.dumps(clean_step_fields(env_result), ensure_ascii=False)

        remaining_steps = [s for s in PIPELINE_STEPS[:-1] if s not in (step1, ENV_STEP)]
        for step in remaining_steps:
            response = _chat_with_retry(lm, model, prompts, step, ctx, existing_names, max_attempts=2)
            if DEBUG and debug:
                debug.log("PIPELINE_STEP_RESULT", f"step={step}\nresponse={response}")
            if not response:
                return None, f"empty response ({step})"
            ctx[step.replace(".txt", "")] = clean_step_json(response)

        assemble_response = ctx["step7_assemble"]
        parsed = extract_json_object(assemble_response or "")
        if DEBUG and debug:
            debug.log("PIPELINE_STEP_RESULT", f"step=step7_assemble\nparsed={parsed}")
        if not isinstance(parsed, dict):
            return None, "step7 invalid json"

        cleaned_parsed = clean_step_fields(parsed)

        parts = _build_parts(cleaned_parsed)

        pos = assemble_positive(parts, context_token=context_token)
        neg = assemble_negative(parts)

        post_process_system = prompts.get("step7_post_process.txt")
        if post_process_system:
            pos, neg = _post_process_with_local_model(lm, model, post_process_system, pos, neg)

        err = validate_result({"prompt": pos, "negative_prompt": neg})
        if DEBUG and debug:
            debug.log("PIPELINE_VALIDATE", f"err={err}\npositive={pos}\nnegative={neg}")
        if err:
            return None, f"validation failed: {err}"

        name_resp = _chat_with_retry(
            lm,
            model,
            prompts,
            "step8_name.txt",
            ctx,
            existing_names,
            max_attempts=2,
        )
        if DEBUG and debug:
            debug.log("PIPELINE_STEP_RESULT", f"step=step8_name\nresponse={name_resp}")

        if not name_resp:
            return None, "empty response (step8)"

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
            if DEBUG and debug:
                debug.log("PIPELINE_STEP_RESULT", f"step=step8_name_retry\nresponse={name_resp}")
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
        if DEBUG and debug:
            debug.log("PIPELINE_RESULT", f"scene_name={scene_name}\nresult={result}")
        return result, None
    finally:
        print(f"[timing] Total: {time.perf_counter() - t0:.2f}s")


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
    debug = get_debug() if DEBUG else None
    if DEBUG and debug:
        debug.log("BATCH_START", f"target={target_count}\nmax_attempts={max_attempts}\nnext_number={next_number}")

    if not data or not any(
        item.get("name") == model and item.get("positive", "") == "" and item.get("negative", "") == ""
        for item in data
    ):
        data.append({"name": model, "positive": "", "negative": ""})

    batch_start = time.perf_counter()
    try:
        while added < target_count and attempts < max_attempts:
            attempts += 1
            existing_names = storage.names(data)
            if on_progress:
                on_progress(added, target_count, attempts)

            result, reason = run_pipeline(lm, model, prompts, existing_names, context_token, data=data)
            if DEBUG and debug:
                debug.log("BATCH_ATTEMPT", f"attempt={attempts}\nadded={added}\nresult={'ok' if result else 'None'}\nreason={reason or 'ok'}")
            if not result:
                print(f"  -> attempt {attempts} failed: {reason}")
                continue

            scene_name = f"{next_number:02d}_{result['_scene_name']}"

            conflict = storage.find_duplicate(
                data,
                name=None,
                prompt=result["prompt"],
            )
            if conflict:
                if DEBUG and debug:
                    debug.log("BATCH_SKIP", f"scene={scene_name}\nreason=duplicate_prompt\nconflict={conflict}")
                print(f"  -> attempt {attempts} skipped: duplicate_prompt ({conflict})")
                continue

            concept = " ".join(
                str(result["_parts"].get(k, ""))
                for k in ("subject", "pose", "state", "environment")
            )
            existing_concepts = storage.prompts(data)
            score, key = lm.max_similarity_with_cache(concept, existing_concepts, cache)
            if not existing_concepts:
                pass
            elif score == 0.0 and key is None:
                if DEBUG and debug:
                    debug.log("BATCH_WARN", f"scene={scene_name}\nembeddings_unavailable")
            elif score > UNIQUENESS_THRESHOLD:
                if DEBUG and debug:
                    debug.log("BATCH_SKIP", f"scene={scene_name}\nreason=too_similar\nscore={score:.2f}")
                print(f"  -> attempt {attempts} skipped: too_similar (score={score:.2f})")
                continue

            data.append({
                "name": scene_name,
                "prompt": result["prompt"],
                "negative_prompt": result["negative_prompt"],
            })
            added += 1
            next_number += 1

            if DEBUG and debug:
                debug.log("BATCH_ADD", f"scene={scene_name}\nadded={added}\nattempts={attempts}")

            if save_path is not None:
                storage.save(save_path, data)
    finally:
        print(f"[timing] Batch total: {time.perf_counter() - batch_start:.2f}s")
        if DEBUG and debug:
            debug.log("BATCH_END", f"added={added}\nattempts={attempts}\ntarget={target_count}")
    return added


def _seed_number(data: list[dict]) -> int:
    from .storage import next_number
    return next_number(data)