from __future__ import annotations

from typing import Any

try:
    import requests
except ModuleNotFoundError:
    requests = None

from .json_utils import cosine_similarity
from .model_info import ModelInfo, parse_v1_model
from .embedding_cache import EmbeddingCache
from .config import (
    LM_STUDIO_URL,
    LM_STUDIO_OPENAI_URL,
    EMBEDDING_MODEL_NAME,
    LM_STUDIO_MODELS_TIMEOUT,
    LM_STUDIO_CHAT_TIMEOUT,
    LM_STUDIO_EMBEDDING_TIMEOUT,
    UNIQUENESS_THRESHOLD,
    DEFAULT_CONTEXT_LENGTH,
    DEBUG,
)
from .debug_log import get as get_debug


class AuthRequired(Exception):
    """Raised when LM Studio returns 401 and no usable token is set."""


class LMClient:
    def __init__(
        self,
        url: str | None = None,
        openai_url: str | None = None,
        token: str | None = None,
    ):
        from .config import LM_STUDIO_URL, LM_STUDIO_OPENAI_URL
        self.url = (url or LM_STUDIO_URL).rstrip("/")
        self.openai_url = (openai_url or LM_STUDIO_OPENAI_URL).rstrip("/")
        if token is not None:
            self.token = token
        else:
            import sys
            from pathlib import Path as _P
            _root = _P(__file__).resolve().parent.parent
            if str(_root) not in sys.path:
                sys.path.insert(0, str(_root))
            import config_loader as _loader
            self.token = _loader.resolve_api_token()
        self._token_attempted = False
        self._models_cache: dict[str, ModelInfo] | None = None
        self._query_embed_cache: EmbeddingCache = EmbeddingCache()
        self.embedding_model_name = EMBEDDING_MODEL_NAME

    def _headers(self) -> dict[str, str]:
        if self.token:
            return {"Authorization": f"Bearer {self.token}"}
        return {}

    def _request(
        self,
        method: str,
        endpoint: str,
        base: str | None = None,
        **kwargs,
    ) -> Any:
        if requests is None:
            raise RuntimeError("requests library is not installed")
        headers = dict(kwargs.pop("headers", {}) or {})
        headers.update(self._headers())
        url = f"{base or self.url}{endpoint}"
        debug = get_debug()
        if DEBUG and debug:
            debug.log("REQUEST", f"{method} {url}\nheaders={headers}\nkwargs={kwargs}")
        try:
            response = requests.request(method, url, headers=headers, **kwargs)
        except Exception as e:
            if DEBUG and debug:
                debug.log("REQUEST_ERROR", f"{method} {url}\nerror={e}")
            raise RuntimeError(f"Network error: {e}") from e
        if response.status_code == 401 and not self._token_attempted:
            raise AuthRequired("LM Studio requires authentication (401).")
        if DEBUG and debug:
            debug.log("RESPONSE", f"{method} {url} -> {response.status_code}\n{response.text[:4000]}")
        response.raise_for_status()
        return response

    def set_token(self, token: str) -> None:
        self.token = token
        self._token_attempted = False

    def _fetch_models(self) -> list[ModelInfo]:
        """Fetch and cache the list of ModelInfo from the server."""
        if self._models_cache is None:
            r = self._request("GET", "/models", timeout=LM_STUDIO_MODELS_TIMEOUT)
            data = r.json().get("models", [])
            self._models_cache = {
                m.get("key", m.get("id", "unknown")): parse_v1_model(m)
                for m in data
            }
        return list(self._models_cache.values())

    def invalidate_models_cache(self) -> None:
        """Force a refresh on the next _fetch_models() call (e.g. after load/unload)."""
        self._models_cache = None

    def list_models_meta(self) -> list[ModelInfo]:
        """Return ModelInfo list from native v1 /api/v1/models."""
        return self._fetch_models()

    def list_chat_models(self) -> list[ModelInfo]:
        return [m for m in self._fetch_models() if m.is_chat]

    def list_models(self) -> list[str]:
        """Return just the names of chat-capable models (legacy convenience)."""
        return [m.key for m in self._fetch_models() if m.is_chat]

    def get_model_info(self, model_name: str) -> ModelInfo | None:
        if self._models_cache is not None:
            cache = self._models_cache
            if model_name in cache:
                return cache[model_name]
            for key, info in cache.items():
                if model_name in key:
                    return info
            return None
        for m in self._fetch_models():
            if m.key == model_name or model_name in m.key:
                return m
        return None

    def is_model_loaded(self, model_name: str) -> bool:
        info = self.get_model_info(model_name)
        if not info:
            return False
        return len(info.loaded_instances) > 0 if hasattr(info, "loaded_instances") else (
            info.loaded_context_length is not None
        )

    def list_loaded_models(self) -> list[ModelInfo]:
        return [m for m in self.list_models_meta() if m.loaded_context_length is not None]

    def unload_all_models(self) -> int:
        """Unload every loaded chat model. Used to free memory before a run.

        Returns the number of models that were unloaded (best effort).
        """
        unloaded = 0
        for m in self.list_loaded_models():
            instance_id = m.key
            loaded_instances = getattr(m, "loaded_instances", None)
            if loaded_instances:
                instance_id = loaded_instances[0].get("id", instance_id)
            if m.loaded_context_length is not None:
                try:
                    self._request(
                        "POST",
                        "/models/unload",
                        json={"instance_id": instance_id},
                        timeout=LM_STUDIO_MODELS_TIMEOUT,
                    )
                    unloaded += 1
                except Exception:
                    pass
        if unloaded:
            self.invalidate_models_cache()
        return unloaded

    def load_model(
        self,
        model_name: str,
        context_length: int | None = None,
    ) -> bool:
        info = self.get_model_info(model_name)
        requested = context_length if context_length is not None else DEFAULT_CONTEXT_LENGTH
        if info:
            requested = info.suggested_context_length(requested)
        try:
            r = self._request(
                "POST",
                "/models/load",
                json={"model": model_name, "context_length": requested},
                timeout=LM_STUDIO_MODELS_TIMEOUT,
            )
            ok = r.json().get("status") == "loaded"
        except Exception:
            return False
        if ok:
            self.invalidate_models_cache()
        return ok

    def chat(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.7,
        max_tokens: int = 600,
        top_k: int | None = None,
        top_p: float | None = None,
        min_p: float | None = None,
        repeat_penalty: float | None = None,
        reasoning: str | None = None,
    ) -> str | None:
        """Send a chat request to LM Studio native v1 endpoint.

        Sampling parameters fall back to architecture-specific presets when
        not provided. Reasoning models get `reasoning="off"` by default
        unless explicitly overridden.
        """
        if requests is None:
            return None

        info = self.get_model_info(model)
        if info:
            preset = info.sampling_preset()
            if temperature == 0.7:
                temperature = preset["temperature"]
            if top_k is None:
                top_k = int(preset["top_k"])
            if top_p is None:
                top_p = float(preset["top_p"])
            if min_p is None:
                min_p = float(preset["min_p"])
            if repeat_penalty is None:
                repeat_penalty = float(preset["repeat_penalty"])
            if reasoning is None and info.is_reasoning_capable and "off" in info.reasoning_allowed:
                reasoning = "off"

        payload: dict[str, Any] = {
            "model": model,
            "system_prompt": system,
            "input": user,
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if top_k is not None:
            payload["top_k"] = top_k
        if top_p is not None:
            payload["top_p"] = top_p
        if min_p is not None:
            payload["min_p"] = min_p
        if repeat_penalty is not None:
            payload["repeat_penalty"] = repeat_penalty
        if reasoning is not None:
            payload["reasoning"] = reasoning

        debug = get_debug()
        if DEBUG and debug:
            debug.log("CHAT_REQUEST", f"model={model}\nsystem={system[:500]}\nuser={user[:500]}\npayload={payload}")

        try:
            r = self._request(
                "POST",
                "/chat",
                json=payload,
                timeout=LM_STUDIO_CHAT_TIMEOUT,
            )
        except Exception as e:
            if DEBUG and debug:
                debug.log("CHAT_ERROR", f"model={model}\nerror={e}")
            return None

        data = r.json()
        for item in data.get("output", []):
            if item.get("type") == "message":
                content = item.get("content")
                if content:
                    if DEBUG and debug:
                        debug.log("CHAT_RESPONSE", f"model={model}\ncontent={content[:4000]}")
                    return content.strip()
        return None

    def embed(
        self,
        inputs: list[str],
        model: str | None = None,
    ) -> list[list[float]] | None:
        """Embeddings via OpenAI-compatible /v1/embeddings."""
        if requests is None:
            return None
        model = model or self.embedding_model_name
        try:
            r = self._request(
                "POST",
                "/embeddings",
                base=self.openai_url,
                json={"model": model, "input": inputs},
                timeout=LM_STUDIO_EMBEDDING_TIMEOUT,
            )
        except Exception:
            return None
        return [
            item.get("embedding")
            for item in r.json().get("data", [])
            if item.get("embedding") is not None
        ]

    def list_embedding_models(self) -> list[str]:
        return [m.key for m in self.list_models_meta() if m.type == "embedding"]

    def has_embedding_model(self, models: list[str] | None = None) -> bool:
        if models is not None:
            return any(self.embedding_model_name in m for m in models)
        for m in self.list_models_meta():
            if m.type == "embedding" and self.embedding_model_name in m.key:
                return True
        return False

    def probe_embedding_model(self) -> bool:
        """Send a tiny probe embedding to confirm the model is loaded.

        Returns True if /v1/embeddings returns a non-empty vector. This
        warms up JIT loading and surfaces missing-model failures early.
        """
        result = self.embed(["probe"], model=self.embedding_model_name)
        return bool(result and any(v for v in result))

    def max_similarity(self, query: str, candidates: list[str]) -> tuple[float, bool]:
        if not candidates:
            return 0.0, False
        vecs = self.embed([query] + candidates)
        if not vecs or len(vecs) != len(candidates) + 1:
            return 0.0, False
        query_vec = vecs[0]
        scores = [cosine_similarity(query_vec, v) for v in vecs[1:]]
        score = max(scores) if scores else 0.0
        return score, score > UNIQUENESS_THRESHOLD

    def max_similarity_with_cache(
        self,
        query: str,
        candidates: list[str],
        cache: EmbeddingCache,
    ) -> tuple[float, str | None]:
        """Compute max similarity using a persistent cache for `candidates`.

        Embeddings for new candidates are fetched in a single batch and
        stored in `cache` so subsequent calls are cheap. The query is
        also cached per-process so repeat-queries (e.g. across retries)
        are free.
        """
        if not candidates:
            return 0.0, None

        def _fetch(missing: list[str]):
            vecs = self.embed(missing)
            return vecs or None

        cand_vecs = cache.ensure(candidates, _fetch)
        if not cand_vecs:
            return 0.0, None

        cached_query = self._query_embed_cache.get(query)
        if cached_query is None:
            query_vecs = self.embed([query])
            if not query_vecs:
                return 0.0, None
            cached_query = query_vecs[0]
            self._query_embed_cache.put(query, cached_query)

        score, key = cache.max_similarity(cached_query, cand_vecs)
        return score, key