"""Embedding backends with a bounded LRU cache and graceful degradation.

Backend priority chain (config `embedder.backend`):

  * fastembed            -> FastEmbedWrapper       (ONNX, lightweight)
  * sentence-transformers-> SentenceTransformerWrapper (PyTorch)
  * hashing              -> HashingEmbedder        (zero-dependency fallback)
  * none                 -> no vectors at all      (BM25-only retrieval)

If the configured backend's module is not importable we degrade to the hashing
backend (deterministic feature-hashing into EMBED_DIM dims, numpy only) unless
the config pins `backend: none`. Heavy model loads stay lazy — the wrapper is
selected at construction, the model downloads / loads on the first embed().

The LRU is hand-rolled (OrderedDict) rather than functools.lru_cache because
the spec calls for a shared mutable cache and lru_cache's keying rules (every
arg must be hashable, cacheable positionally) fight against instance methods.
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from collections import OrderedDict

try:  # the only hard assumption; without numpy every backend degrades to none
    import numpy as np
except ImportError:  # pragma: no cover - environment without numpy
    np = None  # type: ignore[assignment]

from .types import EMBED_DIM

log = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_MAXSIZE = 100


def _module_importable(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


def _token_terms(text: str) -> list[str]:
    """Cheap whitespace/slash split used only for feature hashing."""
    import re

    return re.findall(r"[A-Za-z0-9_一-鿿]+", text.lower())


class Embedder(ABC):
    """Common interface: embed(str) -> np.ndarray (float32)."""

    model: str = DEFAULT_MODEL

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Return a float32 embedding vector for `text`."""


class LRUEmbedder(Embedder):
    """Embedder with a bounded hand-rolled LRU keyed by the exact input text."""

    def __init__(self, model: str = DEFAULT_MODEL, maxsize: int = DEFAULT_MAXSIZE):
        self.model = model
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._maxsize = maxsize

    def embed(self, text: str) -> np.ndarray:
        hit = self._cache.get(text)
        if hit is not None:
            self._cache.move_to_end(text)
            return hit
        vec = self._embed_uncached(text)
        self._cache_put(text, vec)
        return vec

    @abstractmethod
    def _embed_uncached(self, text: str) -> np.ndarray:
        ...

    def _cache_put(self, key: str, value: np.ndarray) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)

    def cache_size(self) -> int:
        return len(self._cache)

    def cache_clear(self) -> None:
        self._cache.clear()


class HashingEmbedder(LRUEmbedder):
    """Deterministic feature-hashing embedding — numpy only, no model weights.

    Maps each token to a signed bucket in EMBED_DIM dims via SHA-256 and
    L2-normalizes. This is a *fallback* (semantically weak) that keeps the
    vector path and RRF working on machines without fastembed / torch; it is
    not a substitute for a real embedding model in production.
    """

    name = "hashing"

    def _embed_uncached(self, text: str) -> np.ndarray:
        vec = np.zeros(EMBED_DIM, dtype=np.float32)
        for term in _token_terms(text):
            digest = hashlib.sha256(term.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "little") % EMBED_DIM
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[bucket] += sign
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec


class FastEmbedWrapper(LRUEmbedder):
    """Primary backend — fastembed (ONNX runtime), lazy model load."""

    name = "fastembed"

    def __init__(self, model: str = DEFAULT_MODEL, maxsize: int = DEFAULT_MAXSIZE):
        super().__init__(model=model, maxsize=maxsize)
        self._fe = None

    def _embed_uncached(self, text: str) -> np.ndarray:
        if self._fe is None:
            from fastembed import TextEmbedding

            self._fe = TextEmbedding(model_name=self.model)
        vec = next(iter(self._fe.embed([text])))
        return np.asarray(vec, dtype=np.float32)


class SentenceTransformerWrapper(LRUEmbedder):
    """Secondary backend — sentence-transformers (PyTorch), lazy model load."""

    name = "sentence-transformers"

    def __init__(self, model: str = DEFAULT_MODEL, maxsize: int = DEFAULT_MAXSIZE):
        super().__init__(model=model, maxsize=maxsize)
        self._st = None

    def _embed_uncached(self, text: str) -> np.ndarray:
        if self._st is None:
            from sentence_transformers import SentenceTransformer

            self._st = SentenceTransformer(self.model)
        return self._st.encode([text], convert_to_numpy=True)[0].astype(np.float32)


# backend -> ordered candidate wrappers (first importable wins).
_BACKEND_CHAIN: dict[str, list[type[LRUEmbedder]]] = {
    "fastembed": [FastEmbedWrapper, SentenceTransformerWrapper, HashingEmbedder],
    "sentence-transformers": [SentenceTransformerWrapper, HashingEmbedder],
    "hashing": [HashingEmbedder],
    "none": [],
}


def get_embedder(config: dict | None = None, *, fallback: bool = True):
    """Resolve an Embedder instance, or None when embeddings are disabled.

    config may carry {'backend': ..., 'model': ...}. The configured chain is
    walked and the first candidate whose module imports is returned; when
    nothing imports and `fallback` is on, the hashing backend is returned.
    """
    config = config or {}
    backend = str(config.get("backend", "fastembed")).strip().lower()
    model = str(config.get("model", DEFAULT_MODEL))
    chain = _BACKEND_CHAIN.get(backend, [HashingEmbedder])
    if not chain:
        return None  # backend: none
    if np is None:
        return None  # every backend needs numpy -> degrade to BM25-only
    for cls in chain:
        if cls is HashingEmbedder:
            if fallback:
                return HashingEmbedder(model=model)
            continue
        # Only select heavy backends whose module actually exists; the model
        # itself stays lazy so a missing network can't break startup.
        if cls is FastEmbedWrapper and not _module_importable("fastembed"):
            continue
        if cls is SentenceTransformerWrapper and not _module_importable(
            "sentence_transformers"
        ):
            continue
        return cls(model=model)
    if fallback:
        return HashingEmbedder(model=model)
    return None
