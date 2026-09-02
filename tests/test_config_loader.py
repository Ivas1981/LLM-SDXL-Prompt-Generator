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

    def test_resolve_uniqueness_threshold_default(self):
        if "UNIQUENESS_THRESHOLD" in os.environ:
            del os.environ["UNIQUENESS_THRESHOLD"]
        cfg_path = ROOT / "config.toml"
        cfg_backup = None
        if cfg_path.exists():
            cfg_backup = cfg_path.read_text(encoding="utf-8")
            cfg_path.write_text("", encoding="utf-8")
        try:
            self.assertEqual(config_loader.resolve_uniqueness_threshold(), 0.85)
        finally:
            if cfg_backup is not None:
                cfg_path.write_text(cfg_backup, encoding="utf-8")

    def test_resolve_uniqueness_threshold_env_overrides_config(self):
        cfg_path = ROOT / "config.toml"
        cfg_exists = cfg_path.exists()
        cfg_backup = None
        if cfg_exists:
            cfg_backup = cfg_path.read_text(encoding="utf-8")
            cfg_path.write_text("[generation]\nuniqueness_threshold = 0.95\n", encoding="utf-8")
        os.environ["UNIQUENESS_THRESHOLD"] = "0.75"
        try:
            self.assertEqual(config_loader.resolve_uniqueness_threshold(), 0.75)
        finally:
            os.environ.pop("UNIQUENESS_THRESHOLD", None)
            if cfg_exists and cfg_backup is not None:
                cfg_path.write_text(cfg_backup, encoding="utf-8")

    def test_resolve_uniqueness_threshold_config_toml(self):
        cfg_path = ROOT / "config.toml"
        cfg_exists = cfg_path.exists()
        cfg_backup = None
        if cfg_exists:
            cfg_backup = cfg_path.read_text(encoding="utf-8")
            cfg_path.write_text("[generation]\nuniqueness_threshold = 0.92\n", encoding="utf-8")
        if "UNIQUENESS_THRESHOLD" in os.environ:
            del os.environ["UNIQUENESS_THRESHOLD"]
        try:
            self.assertEqual(config_loader.resolve_uniqueness_threshold(), 0.92)
        finally:
            if cfg_exists and cfg_backup is not None:
                cfg_path.write_text(cfg_backup, encoding="utf-8")

    def test_resolve_models_timeout_default(self):
        if "MODELS_TIMEOUT" in os.environ:
            del os.environ["MODELS_TIMEOUT"]
        cfg_path = ROOT / "config.toml"
        cfg_backup = None
        if cfg_path.exists():
            cfg_backup = cfg_path.read_text(encoding="utf-8")
            cfg_path.write_text("", encoding="utf-8")
        try:
            self.assertEqual(config_loader.resolve_models_timeout(), 180)
        finally:
            if cfg_backup is not None:
                cfg_path.write_text(cfg_backup, encoding="utf-8")

    def test_resolve_models_timeout_env_overrides_config(self):
        cfg_path = ROOT / "config.toml"
        cfg_exists = cfg_path.exists()
        cfg_backup = None
        if cfg_exists:
            cfg_backup = cfg_path.read_text(encoding="utf-8")
            cfg_path.write_text("[lm_studio]\nmodels_timeout = 240\n", encoding="utf-8")
        os.environ["MODELS_TIMEOUT"] = "300"
        try:
            self.assertEqual(config_loader.resolve_models_timeout(), 300)
        finally:
            os.environ.pop("MODELS_TIMEOUT", None)
            if cfg_exists and cfg_backup is not None:
                cfg_path.write_text(cfg_backup, encoding="utf-8")

    def test_resolve_models_timeout_config_toml(self):
        cfg_path = ROOT / "config.toml"
        cfg_exists = cfg_path.exists()
        cfg_backup = None
        if cfg_exists:
            cfg_backup = cfg_path.read_text(encoding="utf-8")
            cfg_path.write_text("[lm_studio]\nmodels_timeout = 240\n", encoding="utf-8")
        if "MODELS_TIMEOUT" in os.environ:
            del os.environ["MODELS_TIMEOUT"]
        try:
            self.assertEqual(config_loader.resolve_models_timeout(), 240)
        finally:
            if cfg_exists and cfg_backup is not None:
                cfg_path.write_text(cfg_backup, encoding="utf-8")

    def test_resolve_nsfw_default(self):
        if "NSFW" in os.environ:
            del os.environ["NSFW"]
        cfg_path = ROOT / "config.toml"
        cfg_backup = None
        if cfg_path.exists():
            cfg_backup = cfg_path.read_text(encoding="utf-8")
            cfg_path.write_text("", encoding="utf-8")
        try:
            self.assertFalse(config_loader.resolve_nsfw())
        finally:
            if cfg_backup is not None:
                cfg_path.write_text(cfg_backup, encoding="utf-8")

    def test_resolve_nsfw_env_overrides_config(self):
        cfg_path = ROOT / "config.toml"
        cfg_exists = cfg_path.exists()
        cfg_backup = None
        if cfg_exists:
            cfg_backup = cfg_path.read_text(encoding="utf-8")
            cfg_path.write_text("[generation]\nnsfw = false\n", encoding="utf-8")
        os.environ["NSFW"] = "true"
        try:
            self.assertTrue(config_loader.resolve_nsfw())
        finally:
            os.environ.pop("NSFW", None)
            if cfg_exists and cfg_backup is not None:
                cfg_path.write_text(cfg_backup, encoding="utf-8")

    def test_resolve_nsfw_config_toml(self):
        cfg_path = ROOT / "config.toml"
        cfg_exists = cfg_path.exists()
        cfg_backup = None
        if cfg_exists:
            cfg_backup = cfg_path.read_text(encoding="utf-8")
            cfg_path.write_text("[generation]\nnsfw = true\n", encoding="utf-8")
        if "NSFW" in os.environ:
            del os.environ["NSFW"]
        try:
            self.assertTrue(config_loader.resolve_nsfw())
        finally:
            if cfg_exists and cfg_backup is not None:
                cfg_path.write_text(cfg_backup, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
