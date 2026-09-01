import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.embedding_cache import EmbeddingCache


class EmbeddingCacheTests(unittest.TestCase):
    def test_miss_returns_none(self):
        cache = EmbeddingCache()
        self.assertIsNone(cache.get("missing"))

    def test_put_and_get(self):
        cache = EmbeddingCache()
        cache.put("a", [1.0, 0.0])
        self.assertEqual(cache.get("a"), [1.0, 0.0])

    def test_ensure_fetches_missing(self):
        cache = EmbeddingCache()
        called = []

        def _fetch(missing):
            called.extend(missing)
            return [[0.0, 1.0] for _ in missing]

        result = cache.ensure(["x", "y"], _fetch)
        self.assertEqual(called, ["x", "y"])
        self.assertEqual(result, {"x": [0.0, 1.0], "y": [0.0, 1.0]})

    def test_ensure_skips_cached(self):
        cache = EmbeddingCache()
        cache.put("a", [1.0, 0.0])

        def _fetch(missing):
            self.fail("fetch should not be called for cached keys")

        result = cache.ensure(["a"], _fetch)
        self.assertEqual(result, {"a": [1.0, 0.0]})

    def test_max_similarity_returns_best(self):
        cache = EmbeddingCache()
        cache.put("a", [1.0, 0.0])
        cache.put("b", [0.0, 1.0])
        score, key = cache.max_similarity([0.9, 0.1], {"a": [1.0, 0.0], "b": [0.0, 1.0]})
        self.assertAlmostEqual(score, 0.99, places=2)
        self.assertEqual(key, "a")


if __name__ == "__main__":
    unittest.main()
