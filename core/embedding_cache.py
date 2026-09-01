from __future__ import annotations

from .json_utils import cosine_similarity


class EmbeddingCache:
    """Keeps embeddings for already-seen strings to avoid re-querying LM Studio.

    Lookup is by exact string key. New keys are computed in a single batch
    via the supplied client callback. The cache is append-only: removed
    keys cannot be detected, so callers should not rely on invalidation.
    """

    def __init__(self):
        self._store: dict[str, list[float]] = {}

    def __contains__(self, key: str) -> bool:
        return key in self._store

    def __len__(self) -> int:
        return len(self._store)

    def get(self, key: str) -> list[float] | None:
        return self._store.get(key)

    def put(self, key: str, vector: list[float]) -> None:
        self._store[key] = vector

    def ensure(
        self,
        keys: list[str],
        embed_fn,
    ) -> dict[str, list[float]]:
        """Ensure every key has an entry; compute missing ones via embed_fn.

        `embed_fn(list[str]) -> list[list[float] | None]` is called once
        with only the missing keys. Returns the full mapping.
        """
        missing = [k for k in keys if k not in self._store]
        if missing:
            results = embed_fn(missing)
            for k, v in zip(missing, results):
                if v is not None:
                    self._store[k] = v
        return {k: self._store[k] for k in keys if k in self._store}

    def max_similarity(self, query_vec: list[float], candidates: dict[str, list[float]]) -> tuple[float, str | None]:
        best_score = 0.0
        best_key: str | None = None
        for key, vec in candidates.items():
            score = cosine_similarity(query_vec, vec)
            if score > best_score:
                best_score = score
                best_key = key
        return best_score, best_key