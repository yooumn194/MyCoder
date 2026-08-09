"""Phase 5 embedder: multi-backend resolution + bounded LRU cache (spec: 4)."""

import numpy as np

from corecoder.memory.embedder import (
    DEFAULT_MODEL,
    FastEmbedWrapper,
    HashingEmbedder,
    _module_importable,
    get_embedder,
)


def test_hashing_embedder_deterministic_and_normalized():
    emb = HashingEmbedder()
    v1 = emb.embed("认证模块使用JWT")
    v2 = emb.embed("认证模块使用JWT")
    assert v1.shape == (512,)
    assert np.allclose(v1, v2)  # deterministic across calls
    assert np.isclose(np.linalg.norm(v1), 1.0, atol=1e-3)  # L2-normalized


def test_lru_cache_is_bounded_and_reuses_objects():
    emb = HashingEmbedder(maxsize=5)
    for i in range(20):
        emb.embed(f"文档片段内容编号 {i}")
    assert emb.cache_size() <= 5  # evicted beyond maxsize

    first = emb.embed("重复文本片段")
    second = emb.embed("重复文本片段")
    assert first is second  # identical object from the cache

    emb.cache_clear()
    assert emb.cache_size() == 0


def test_get_embedder_backend_selection():
    assert isinstance(get_embedder({"backend": "hashing"}), HashingEmbedder)
    # unknown backend degrades to hashing (never raises)
    assert isinstance(get_embedder({"backend": "bogus"}), HashingEmbedder)
    # backend: none disables vectors entirely
    assert get_embedder({"backend": "none"}) is None


def test_get_embedder_heavy_backend_selection():
    # Deterministic given the installed environment:
    #   fastembed present  -> FastEmbedWrapper
    #   fastembed absent   -> fallback=None when fallback is disabled
    if _module_importable("fastembed"):
        assert isinstance(
            get_embedder({"backend": "fastembed", "model": DEFAULT_MODEL}, fallback=False),
            FastEmbedWrapper,
        )
    else:
        assert (
            get_embedder({"backend": "fastembed"}, fallback=False) is None
        )
    # with fallback enabled the hashing backend is always reachable
    assert isinstance(get_embedder({"backend": "fastembed"}), HashingEmbedder)
