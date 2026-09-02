import unittest
from pathlib import Path
import sys
import os

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config_loader


class ConfigLoaderTests(unittest.TestCase):
    def test_resolve_url_default(self):
        url = config_loader.resolve_url()
        self.assertIn("http", url)

    def test_resolve_openai_url_default(self):
        url = config_loader.resolve_openai_url()
        self.assertIn("http", url)

    def test_resolve_api_token_default_none(self):
        token = config_loader.resolve_api_token()
        if os.environ.get("LM_API_TOKEN") or (ROOT / "config.toml").exists():
            self.assertIsNotNone(token)
        else:
            self.assertIsNone(token)

    def test_config_path_returns_none_when_missing(self):
        path = config_loader.config_path()
        if (ROOT / "config.toml").exists():
            self.assertIsNotNone(path)
        else:
            self.assertIsNone(path)

    def test_resolve_debug_from_config_toml(self):
        debug = config_loader.resolve_debug()
        if (ROOT / "config.toml").exists():
            cfg = config_loader._load_toml(ROOT / "config.toml")
            v = config_loader._get(cfg, "lm_studio", "debug")
            if isinstance(v, bool):
                self.assertEqual(debug, v)
            elif isinstance(v, str):
                self.assertEqual(debug, v.lower() in ("on", "1", "true", "yes"))
            else:
                self.assertFalse(debug)
        else:
            self.assertFalse(debug)

    def test_resolve_debug_env_overrides_config(self):
        cfg_path = ROOT / "config.toml"
        cfg_exists = cfg_path.exists()
        cfg_backup = None
        if cfg_exists:
            cfg_backup = cfg_path.read_text(encoding="utf-8")
            cfg_path.write_text("[lm_studio]\ndebug = false\n", encoding="utf-8")
        os.environ["DEBUG"] = "on"
        try:
            debug = config_loader.resolve_debug()
            self.assertTrue(debug)
        finally:
            os.environ.pop("DEBUG", None)
            if cfg_exists and cfg_backup is not None:
                cfg_path.write_text(cfg_backup, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
