import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROMPTS_DIR = BASE_DIR / "prompts"

# config_loader lives at the project root. The core/ package may be imported
# either as `core.config` (when tests/prepend sys.path) or as a plain
# `config` module (when run from inside core/). Use an absolute import in
# both cases by inserting the root directory.
_root_for_loader = BASE_DIR
if str(_root_for_loader) not in sys.path:
    sys.path.insert(0, str(_root_for_loader))
import config_loader as _loader  # noqa: E402
import env_registry as _env  # noqa: E402

LM_STUDIO_URL = _loader.resolve_url()
LM_STUDIO_OPENAI_URL = _loader.resolve_openai_url()

EMBEDDING_MODEL_NAME = _env.get_str("EMBEDDING_MODEL_NAME")

_output_json = _env.get_str("OUTPUT_JSON")
JSON_FILE = Path(_output_json) if _output_json else BASE_DIR / "sdxl_styles.json"

UNIQUENESS_THRESHOLD = _env.get_float("UNIQUENESS_THRESHOLD")
MAX_ATTEMPTS_MULTIPLIER = _env.get_int("MAX_ATTEMPTS_MULTIPLIER")

LM_STUDIO_MODELS_TIMEOUT = 30
LM_STUDIO_CHAT_TIMEOUT = _env.get_int("CHAT_TIMEOUT")
LM_STUDIO_EMBEDDING_TIMEOUT = 120

PIPELINE_STEPS = (
    "step1_concept.txt",
    "step2_environment.txt",
    "step3_pose.txt",
    "step4_state.txt",
    "step5_lighting.txt",
    "step6_camera.txt",
    "step7_assemble.txt",
    "step8_name.txt",
)

DEFAULT_CONTEXT_TOKEN = _env.get_str("CONTEXT_TOKEN")
FORBIDDEN_TAGS = (
    "woman", "girl", "female", "lady", "person", "human",
    "blonde", "brunette", "redhead", "black hair", "brown hair", "dark hair", "light hair",
    "slim", "curvy", "athletic", "thin", "fat", "skinny", "overweight",
    "tall", "short", "petite", "young", "old", "elderly", "teen", "adult",
    "asian", "european", "african", "caucasian", "latino",
)

FORBIDDEN_TAGS_NEGATIVE: tuple[str, ...] = ()

NEGATIVE_BASE = _env.get_str("NEGATIVE_BASE_TAGS")

QUALITY_BAIT_TAGS = (
    "masterpiece", "best quality", "worst quality", "high quality", "low quality",
    "ultra detailed", "highly detailed", "intricate", "sharp focus",
    "8k", "16k", "4k", "32k", "absurdres",
    "hdr", "uhd", "professional", "perfect",
    "beautiful", "gorgeous", "stunning", "amazing", "incredible",
)

RECENT_NAMES_CONTEXT = 30

DEFAULT_CONTEXT_LENGTH = _env.get_int("LM_CONTEXT_LENGTH")

ARCH_PRESETS: dict[str, dict[str, float]] = {
    "qwen3": {"temperature": 0.7, "top_k": 20, "top_p": 0.8, "min_p": 0.0, "repeat_penalty": 1.05},
    "qwen3vl": {"temperature": 0.7, "top_k": 20, "top_p": 0.8, "min_p": 0.0, "repeat_penalty": 1.05},
    "qwen3moe": {"temperature": 0.6, "top_k": 20, "top_p": 0.9, "min_p": 0.05, "repeat_penalty": 1.05},
    "qwen35": {"temperature": 0.7, "top_k": 20, "top_p": 0.8, "min_p": 0.0, "repeat_penalty": 1.05},
    "qwen35moe": {"temperature": 0.7, "top_k": 20, "top_p": 0.8, "min_p": 0.0, "repeat_penalty": 1.05},
    "qwen2": {"temperature": 0.7, "top_k": 20, "top_p": 0.8, "min_p": 0.0, "repeat_penalty": 1.05},
    "gpt-oss": {"temperature": 0.7, "top_k": 40, "top_p": 0.95, "min_p": 0.05, "repeat_penalty": 1.0},
    "mistral3": {"temperature": 0.7, "top_k": 40, "top_p": 0.9, "min_p": 0.05, "repeat_penalty": 1.05},
    "gemma4": {"temperature": 0.7, "top_k": 40, "top_p": 0.95, "min_p": 0.05, "repeat_penalty": 1.05},
    "gemma3": {"temperature": 0.7, "top_k": 40, "top_p": 0.95, "min_p": 0.05, "repeat_penalty": 1.05},
    "deepseek2": {"temperature": 0.7, "top_k": 40, "top_p": 0.9, "min_p": 0.05, "repeat_penalty": 1.05},
    "nemotron_h": {"temperature": 0.7, "top_k": 40, "top_p": 0.9, "min_p": 0.05, "repeat_penalty": 1.05},
    "_default": {"temperature": 0.7, "top_k": 40, "top_p": 0.9, "min_p": 0.0, "repeat_penalty": 1.05},
}