from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import config
from core.lm_client import LMClient, AuthRequired
from core.pipeline import generate_batch, load_system_prompts
from core import storage
import env_registry


def prompt_int(question: str, lo: int, hi: int) -> int:
    while True:
        raw = input(question).strip()
        try:
            value = int(raw)
            if lo <= value <= hi:
                return value
            print(f"Please enter a number between {lo} and {hi}.")
        except ValueError:
            print("Please enter a valid integer.")


def prompt_token() -> str | None:
    try:
        import getpass
        token = getpass.getpass("LM Studio API token (input hidden): ").strip()
    except Exception:
        token = input("LM Studio API token: ").strip()
    return token or None


def select_model(models: list[str]) -> str:
    print("\n" + "=" * 60)
    print("AVAILABLE MODELS")
    print("=" * 60)
    for i, m in enumerate(models, 1):
        label = m if len(m) <= 50 else m[:47] + "..."
        print(f"  [{i:2d}] {label}")
    print("=" * 60)
    idx = prompt_int("\nModel number: ", 1, len(models))
    return models[idx - 1]


def connect_with_retry() -> LMClient:
    lm = LMClient()
    while True:
        try:
            models = lm.list_models()
            return lm, models
        except AuthRequired:
            print("LM Studio requires authentication (401).")
            token = prompt_token()
            if not token:
                print("No token provided. Aborting.")
                sys.exit(1)
            lm.set_token(token)
            print("Token saved for this session. Retrying...")
        except Exception as e:
            print(f"Cannot connect to LM Studio: {e}")
            sys.exit(1)


def run() -> int:
    print("\n" + "=" * 60)
    print("SDXL PROMPT GENERATOR (refactored)")
    print("=" * 60)

    from config_loader import config_path
    cfg_path = config_path()
    if cfg_path:
        print(f"Using config: {cfg_path}")
    else:
        print("No config.toml found, using env / defaults.")

    prompts_dir = config.PROMPTS_DIR
    if not prompts_dir.exists():
        print(f"Prompts directory not found: {prompts_dir}")
        return 1

    print("\nLoading pipeline prompts...")
    prompts = load_system_prompts(prompts_dir)
    print(f"Loaded {len(prompts)} system prompts.")

    print("\nConnecting to LM Studio...")
    lm, models = connect_with_retry()

    if not models:
        print("No models available in LM Studio.")
        return 1

    print("\nUnloading any previously loaded models to free resources...")
    try:
        unloaded = lm.unload_all_models()
        if unloaded:
            print(f"Unloaded {unloaded} model(s).")
        else:
            print("No models were loaded.")
    except Exception as e:
        print(f"Warning: unload failed: {e}")

    if lm.has_embedding_model():
        print("Probing embedding model (may trigger JIT load)...")
        if lm.probe_embedding_model():
            print("Embedding model is responsive.")
        else:
            print("Warning: embedding model probe returned empty. Duplicate checks may silently degrade.")
    else:
        print(f"Warning: embedding model '{config.EMBEDDING_MODEL_NAME}' not found among downloaded models.")
        print("Semantic duplicate checks will be skipped; exact-name and exact-prompt checks still apply.")

    model_name = select_model(models)

    try:
        if not lm.is_model_loaded(model_name):
            if lm.load_model(model_name):
                print(f"Loaded model {model_name}")
            else:
                print(f"Failed to load {model_name}. Aborting.")
                return 1
        else:
            print(f"Model {model_name} is already loaded.")
    except AuthRequired:
        token = prompt_token()
        if not token:
            print("No token provided. Aborting.")
            return 1
        lm.set_token(token)
        if lm.load_model(model_name):
            print(f"Loaded model {model_name}")
        else:
            print(f"Failed to load {model_name} after auth. Aborting.")
            return 1

    target = prompt_int("\nHow many scenes to generate? (1-1000): ", 1, 1000)
    context_token = os.environ.get("CONTEXT_TOKEN", config.DEFAULT_CONTEXT_TOKEN)

    data = storage.load_or_init(config.JSON_FILE)
    initial = len(data)
    print(f"Existing entries: {initial} (next index: {initial + 1})")

    print(f"\nGenerating {target} new prompts (max attempts = {target * config.MAX_ATTEMPTS_MULTIPLIER})...")
    print("Press Ctrl+C to stop safely.\n")

    def progress_cb(a_idx: int, total: int, attempt: int) -> None:
        print(f"[{a_idx}/{total}] attempt #{attempt} ...")

    try:
        added = generate_batch(
            lm=lm,
            model=model_name,
            prompts=prompts,
            data=data,
            target_count=target,
            context_token=context_token,
            on_progress=progress_cb,
            save_path=config.JSON_FILE,
        )
    except KeyboardInterrupt:
        storage.save(config.JSON_FILE, data)
        print("\nInterrupted. Progress saved.")
        return 0

    storage.save(config.JSON_FILE, data)
    print("\n" + "=" * 60)
    if added >= target:
        print(f"Done. Added {target} prompts. Total: {len(data)}.")
    else:
        print(f"Stopped early after attempts. Added {added}/{target}. Total: {len(data)}.")
    print(f"Output file: {config.JSON_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(run())
