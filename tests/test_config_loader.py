"""Tests for config_loader module."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config_loader


_ENV_KEYS = (
    "LM_STUDIO_URL", "LM_STUDIO_OPENAI_URL", "LM_API_TOKEN",
    "PROMPTGEN_CONFIG", "DEBUG", "NSFW", "UNIQUENESS_THRESHOLD",
    "MODELS_TIMEOUT", "CHAT_TIMEOUT",
)


class ConfigLoaderTests(unittest.TestCase):
    def setUp(self):
        self._saved_env = {k: os.environ[k] for k in _ENV_KEYS if k in os.environ}
        for k in _ENV_KEYS:
            os.environ.pop(k, None)
        self._tmp = tempfile.mkdtemp()
        self._cfg_path = Path(self._tmp) / "config.toml"

    def tearDown(self):
        for k in _ENV_KEYS:
            os.environ.pop(k, None)
        for k, v in self._saved_env.items():
            os.environ[k] = v

    def _use_empty_config(self):
        os.environ["PROMPTGEN_CONFIG"] = str(self._cfg_path)
        self._cfg_path.write_text("", encoding="utf-8")

    def _use_config(self, content: str):
        os.environ["PROMPTGEN_CONFIG"] = str(self._cfg_path)
        self._cfg_path.write_text(content, encoding="utf-8")

    def test_resolve_url_default(self):
        url = config_loader.resolve_url()
        self.assertIn("http", url)
        self.assertIn("1234", url)

    def test_resolve_openai_url_default(self):
        url = config_loader.resolve_openai_url()
        self.assertIn("http", url)
        self.assertIn("/v1", url)

    def test_resolve_api_token_default_none(self):
        self._use_empty_config()
        self.assertIsNone(config_loader.resolve_api_token())

    def test_resolve_api_token_from_env(self):
        os.environ["LM_API_TOKEN"] = "abc"
        self._use_empty_config()
        self.assertEqual(config_loader.resolve_api_token(), "abc")

    def test_resolve_api_token_from_config(self):
        os.environ.pop("LM_API_TOKEN", None)
        self._use_config('[lm_studio]\napi_token = "from_toml"\n')
        self.assertEqual(config_loader.resolve_api_token(), "from_toml")

    def test_config_path_returns_none_when_missing(self):
        self._use_empty_config()
        self._cfg_path.unlink()
        self.assertIsNone(config_loader.config_path())

    def test_resolve_debug_default(self):
        self._use_empty_config()
        self.assertFalse(config_loader.resolve_debug())

    def test_resolve_debug_env_overrides_config(self):
        self._use_config("[lm_studio]\ndebug = false\n")
        os.environ["DEBUG"] = "on"
        self.assertTrue(config_loader.resolve_debug())

    def test_resolve_debug_from_config(self):
        self._use_config("[lm_studio]\ndebug = true\n")
        self.assertTrue(config_loader.resolve_debug())

    def test_resolve_uniqueness_threshold_default(self):
        self._use_empty_config()
        self.assertEqual(config_loader.resolve_uniqueness_threshold(), 0.85)

    def test_resolve_uniqueness_threshold_env_overrides_config(self):
        self._use_config("[generation]\nuniqueness_threshold = 0.95\n")
        os.environ["UNIQUENESS_THRESHOLD"] = "0.75"
        self.assertEqual(config_loader.resolve_uniqueness_threshold(), 0.75)

    def test_resolve_uniqueness_threshold_config_toml(self):
        self._use_config("[generation]\nuniqueness_threshold = 0.92\n")
        self.assertEqual(config_loader.resolve_uniqueness_threshold(), 0.92)

    def test_resolve_models_timeout_default(self):
        self._use_empty_config()
        self.assertEqual(config_loader.resolve_models_timeout(), 180)

    def test_resolve_models_timeout_env_overrides_config(self):
        self._use_config("[lm_studio]\nmodels_timeout = 240\n")
        os.environ["MODELS_TIMEOUT"] = "300"
        self.assertEqual(config_loader.resolve_models_timeout(), 300)

    def test_resolve_models_timeout_config_toml(self):
        self._use_config("[lm_studio]\nmodels_timeout = 240\n")
        self.assertEqual(config_loader.resolve_models_timeout(), 240)

    def test_resolve_nsfw_default(self):
        self._use_empty_config()
        self.assertFalse(config_loader.resolve_nsfw())

    def test_resolve_nsfw_env_overrides_config(self):
        self._use_config("[generation]\nnsfw = false\n")
        os.environ["NSFW"] = "true"
        self.assertTrue(config_loader.resolve_nsfw())

    def test_resolve_nsfw_config_toml(self):
        self._use_config("[generation]\nnsfw = true\n")
        self.assertTrue(config_loader.resolve_nsfw())

    def test_resolve_url_from_config(self):
        self._use_config('[lm_studio]\nurl = "http://10.0.0.5:9000/api/v1"\n')
        self.assertEqual(config_loader.resolve_url(), "http://10.0.0.5:9000/api/v1")

    def test_resolve_openai_url_derived_from_native(self):
        self._use_config('[lm_studio]\nurl = "http://localhost:1234/api/v1"\n')
        self.assertEqual(config_loader.resolve_openai_url(), "http://localhost:1234/v1")


if __name__ == "__main__":
    unittest.main()