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

    def test_resolve_debug_default_false(self):
        debug = config_loader.resolve_debug()
        self.assertFalse(debug)


if __name__ == "__main__":
    unittest.main()
