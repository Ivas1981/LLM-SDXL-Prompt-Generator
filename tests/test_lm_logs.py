import unittest
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import lm_logs


class LMLogsTests(unittest.TestCase):
    def test_latest_log_path_returns_path_or_none(self):
        # Test with non-existent directory
        with tempfile.TemporaryDirectory() as tmp:
            path = lm_logs.latest_log_path(Path(tmp))
            self.assertIsNone(path)

        # Test with existing directory but no log files
        with tempfile.TemporaryDirectory() as tmp:
            path = lm_logs.latest_log_path(Path(tmp))
            self.assertIsNone(path)

        # Test with existing log file
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            log_file = log_dir / "test.log"
            log_file.write_text("test")
            path = lm_logs.latest_log_path(log_dir)
            self.assertIsInstance(path, Path)
            self.assertEqual(path, log_file)


if __name__ == "__main__":
    unittest.main()
