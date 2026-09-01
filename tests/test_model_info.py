import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.model_info import ModelInfo, parse_v1_model


class ModelInfoTests(unittest.TestCase):
    def test_sampling_preset_qwen3vl(self):
        info = ModelInfo(
            key="m", type="llm", architecture="qwen3vl",
            quantization="Q4_K_S", bits_per_weight=4,
            max_context_length=8192, params_string="4B", size_bytes=1,
            vision=False, tool_use=True,
        )
        preset = info.sampling_preset()
        self.assertEqual(preset["temperature"], 0.7)
        self.assertEqual(preset["top_k"], 20)

    def test_sampling_preset_gemma4(self):
        info = ModelInfo(
            key="m", type="llm", architecture="gemma4",
            quantization="Q4_K_S", bits_per_weight=4,
            max_context_length=8192, params_string="4B", size_bytes=1,
            vision=False, tool_use=True,
        )
        preset = info.sampling_preset()
        self.assertEqual(preset["top_k"], 40)
        self.assertEqual(preset["top_p"], 0.95)

    def test_parse_v1_model_basic(self):
        raw = {
            "key": "test-model",
            "type": "llm",
            "architecture": "qwen3",
            "quantization": {"name": "Q4_K_S", "bits_per_weight": 4},
            "max_context_length": 4096,
            "params_string": "4B",
            "size_bytes": 1000,
            "capabilities": {"vision": False, "trained_for_tool_use": True},
        }
        info = parse_v1_model(raw)
        self.assertEqual(info.key, "test-model")
        self.assertEqual(info.architecture, "qwen3")
        self.assertTrue(info.is_chat)
        self.assertFalse(info.vision)

    def test_is_reasoning_capable(self):
        info = ModelInfo(
            key="m", type="llm", architecture="qwen3",
            quantization="Q4_K_S", bits_per_weight=4,
            max_context_length=8192, params_string="4B", size_bytes=1,
            vision=False, tool_use=True,
            reasoning_allowed=["off", "on"],
            reasoning_default="on",
        )
        self.assertTrue(info.is_reasoning_capable)


if __name__ == "__main__":
    unittest.main()
