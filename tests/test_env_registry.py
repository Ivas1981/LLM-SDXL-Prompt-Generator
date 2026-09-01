import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import env_registry


class EnvRegistryTests(unittest.TestCase):
    def test_get_str_returns_value(self):
        import os
        os.environ["LM_STUDIO_URL"] = "http://localhost:1234/api/v1"
        try:
            val = env_registry.get_str("LM_STUDIO_URL")
            self.assertEqual(val, "http://localhost:1234/api/v1")
        finally:
            del os.environ["LM_STUDIO_URL"]

    def test_get_int_returns_value(self):
        import os
        os.environ["CHAT_TIMEOUT"] = "120"
        try:
            val = env_registry.get_int("CHAT_TIMEOUT")
            self.assertEqual(val, 120)
        finally:
            del os.environ["CHAT_TIMEOUT"]

    def test_get_float_returns_value(self):
        import os
        os.environ["UNIQUENESS_THRESHOLD"] = "0.9"
        try:
            val = env_registry.get_float("UNIQUENESS_THRESHOLD")
            self.assertEqual(val, 0.9)
        finally:
            del os.environ["UNIQUENESS_THRESHOLD"]

    def test_describe_contains_known_keys(self):
        entries = env_registry.describe()
        names = [e["name"] for e in entries]
        self.assertIn("LM_STUDIO_URL", names)
        self.assertIn("EMBEDDING_MODEL_NAME", names)

    def test_all_names_returns_list(self):
        names = env_registry.all_names()
        self.assertIsInstance(names, list)
        self.assertIn("LM_STUDIO_URL", names)


if __name__ == "__main__":
    unittest.main()
