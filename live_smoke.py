"""Live smoke test: run all 8 pipeline steps against the real LM Studio server.

This script is intentionally separate from main.py and from the unittest
suite. It exercises the real network path end-to-end so we can inspect the
model output. It does NOT save to sdxl_styles.json (uses a throwaway file).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.lm_client import LMClient
from core import config
from core.pipeline import load_system_prompts, run_pipeline


OUTPUT_PATH = Path("smoke_test_output.json")


def main(model_name: str | None = None) -> int:
    lm = LMClient()

    print("=" * 60)
    print("LIVE SMOKE TEST — full 8-step pipeline")
    print("=" * 60)

    print("\nUnloading all models...")
    unloaded = lm.unload_all_models()
    print(f"Unloaded {unloaded} model(s).")

    if model_name is None:
        model_name = "qwen3-vl-4b-instruct-uncensored-i1"
    print(f"Selected model: {model_name}")

    info = lm.get_model_info(model_name)
    if info is None:
        print(f"Model {model_name} not found.")
        return 1

    print(f"  arch={info.architecture}, max_ctx={info.max_context_length}, "
          f"params={info.params_string}, vision={info.vision}")

    suggested = info.suggested_context_length(config.DEFAULT_CONTEXT_LENGTH)
    print(f"Loading with context_length={suggested}...")
    if not lm.load_model(model_name, context_length=suggested):
        print("Failed to load model.")
        return 1
    print("Model loaded.")

    prompts = load_system_prompts(config.PROMPTS_DIR)
    print(f"Loaded {len(prompts)} system prompts.")

    existing_names: list[str] = []
    print("\nRunning pipeline...")
    t0 = time.time()

    # Replicate pipeline with logging to find the failing step.
    from core import pipeline as P

    ctx: dict[str, str] = {}
    step1 = "step1_concept.txt"
    temp, mt = P.STEP_PARAMS[step1]
    resp = lm.chat(model_name, prompts[step1],
                   P._format_user_hint(step1, ctx, existing_names),
                   temperature=temp, max_tokens=mt)
    print(f"  step1: ok={bool(resp)}, len={len(resp) if resp else 0}")
    if not resp:
        return 1
    ctx[step1.replace(".txt", "")] = resp

    env_result = P._run_environment_step(lm, model_name, prompts, ctx, existing_names)
    print(f"  step2: env_ok={bool(env_result)}")
    if not env_result:
        return 1
    ctx["step2_environment"] = json.dumps(env_result, ensure_ascii=False)

    for step in [s for s in P.PIPELINE_STEPS[:-1] if s not in (step1, "step2_environment.txt")]:
        temp, mt = P.STEP_PARAMS[step]
        resp = lm.chat(model_name, prompts[step],
                       P._format_user_hint(step, ctx, existing_names),
                       temperature=temp, max_tokens=mt)
        print(f"  {step}: ok={bool(resp)}, len={len(resp) if resp else 0}")
        if not resp:
            return 1
        ctx[step.replace(".txt", "")] = resp

    assemble_response = ctx["step7_assemble"]
    parsed = P.extract_json_object(assemble_response or "")
    print(f"  parse step7: dict={isinstance(parsed, dict)}")
    if not isinstance(parsed, dict):
        print(f"  raw assemble response[:300]: {assemble_response[:300]}")
        return 1

    parts = {
        "subject": parsed.get("subject") or "",
        "pose": parsed.get("pose") or "",
        "state": parsed.get("state") or "",
        "environment": parsed.get("environment") or "",
        "relationships": parsed.get("relationships") or "",
        "lighting": parsed.get("lighting") or "",
        "camera": parsed.get("camera") or "",
    }
    if not parts["subject"]:
        print(f"  step7 keys: {list(parsed.keys())}")
        print(f"  raw parsed: {parsed}")
    pos = P.assemble_positive(parts)
    print(f"  pos has token: {config.DEFAULT_CONTEXT_TOKEN in pos}, len={len(pos)}")
    print(f"  pos[:200]: {pos[:200]}")
    neg = P.assemble_negative(parts)
    err = P.validate_result({"prompt": pos, "negative_prompt": neg})
    print(f"  validate: err={err}")
    if err:
        return 1

    name_resp = lm.chat(model_name, prompts["step8_name.txt"],
                       P._format_user_hint("step8_name.txt", ctx, existing_names),
                       temperature=P.STEP_PARAMS["step8_name.txt"][0],
                       max_tokens=P.STEP_PARAMS["step8_name.txt"][1])
    print(f"  step8: ok={bool(name_resp)}, raw={name_resp[:60] if name_resp else None}")
    if not name_resp:
        return 1

    from core.pipeline import _safe_name
    scene_name = _safe_name(name_resp or "")
    print(f"  scene_name: {scene_name}")

    result = {
        "prompt": pos,
        "negative_prompt": neg,
        "_parts": parts,
        "_scene_name": scene_name,
        "_raw_assembled": {
            "prompt": parsed.get("prompt", ""),
            "negative_prompt": parsed.get("negative_prompt", ""),
        },
    }
    elapsed = time.time() - t0
    print(f"\nManual pipeline OK in {elapsed:.1f}s")
    print("\n--- _parts (raw fields from step7) ---")
    for k, v in result["_parts"].items():
        print(f"  {k}: {v}")

    print("\n--- Final positive prompt ---")
    print(result["prompt"])

    print("\n--- Final negative prompt ---")
    print(result["negative_prompt"])

    print(f"\n--- Scene name: {result['_scene_name']}")

    print("\n--- Raw step7 response (truncated) ---")
    raw = result["_raw_assembled"]
    print(json.dumps(raw, ensure_ascii=False, indent=2))

    OUTPUT_PATH.write_text(
        json.dumps(
            {
                "model": model_name,
                "elapsed_s": round(elapsed, 2),
                "name": result["_scene_name"],
                "prompt": result["prompt"],
                "negative_prompt": result["negative_prompt"],
                "parts": result["_parts"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    sys.exit(main(arg))