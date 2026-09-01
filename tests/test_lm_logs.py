import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import lm_logs


class LMLogsTests(unittest.TestCase):
    def test_latest_log_path_returns_path(self):
        path = lm_logs.latest_log_path()
        self.assertIsInstance(path, Path)


if __name__ == "__main__":
    unittest.main()
