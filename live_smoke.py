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

    result, reason = run_pipeline(
        lm=lm,
        model=model_name,
        prompts=prompts,
        existing_names=existing_names,
        data=[],
    )

    elapsed = time.time() - t0
    if not result:
        print(f"Pipeline failed: {reason}")
        return 1

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