"""Phase 5 hybrid retriever: BM25 (Chinese word-level), vectors, RRF fusion,
filters (spec: 10 tests)."""

import pytest

from corecoder.memory import MemoryEntry
from corecoder.memory.retriever import HybridRetriever


def _save(memory_store, content, **kw):
    return memory_store.save(MemoryEntry(content=content, **kw))


def test_bm25_chinese_word_level_match(memory_store):
    """Acceptance: '认证模块' matches '认证模块使用JWT...' via word-level
    tokens (jieba), not single characters."""
    _save(memory_store, "认证模块使用JWT进行身份验证", type="project")
    _save(memory_store, "日志系统按天轮转", type="fact")
    scored = memory_store.bm25_search("认证模块", limit=5)
    assert scored
    assert memory_store.get(scored[0][0]).content == "认证模块使用JWT进行身份验证"


def test_bm25_ranking_prefers_higher_term_frequency(memory_store):
    _save(memory_store, "缓存 设计 缓存 设计 缓存 设计")
    _save(memory_store, "缓存 设计")
    scored = memory_store.bm25_search("缓存 设计")
    assert scored
    assert memory_store.get(scored[0][0]).content.startswith("缓存 设计 缓存")


def test_bm25_scores_normalized_to_unit_range(memory_store):
    _save(memory_store, "JWT JWT 令牌")
    _save(memory_store, "JWT 鉴权")
    scored = memory_store.bm25_search("JWT")
    assert scored
    # FTS5 raw rank is negative; normalized scores live in [0, 1]
    assert all(0.0 <= score <= 1.0 for _, score in scored)
    # higher term frequency ranks first
    assert memory_store.get(scored[0][0]).content == "JWT JWT 令牌"


def test_rrf_fuse_combines_ranked_lists():
    retriever = object.__new__(HybridRetriever)
    retriever.rrf_k = 60
    a = [("idA", 1.0), ("idB", 0.8)]
    b = [("idB", 0.9), ("idC", 0.7)]
    fused = retriever._rrf_fuse(a, b)
    ids = [mem_id for mem_id, _ in fused]
    # idB: 1/61 (list a, rank 2) + 1/62 (list b, rank 1); idA: 1/61; idC: 1/62
    assert ids == ["idB", "idA", "idC"]
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)


def test_hybrid_returns_results_when_bm25_empty(memory_store, memory_retriever):
    """No BM25 token overlap -> the vector path still returns candidates."""
    _save(memory_store, "数据备份策略说明")
    hits = memory_retriever.search("日志轮转", limit=5)
    assert hits  # vector path (hashed) keeps hybrid retrieval alive
    assert "数据备份策略说明" in {h["content"] for h in hits}


def test_search_filters_by_scope(memory_store, memory_retriever):
    _save(memory_store, "项目本地记忆", scope="project")
    _save(memory_store, "全局记忆内容", scope="global")
    hits = memory_retriever.search("记忆", scope="project", limit=5)
    assert hits
    assert all(h["scope"] == "project" for h in hits)


def test_search_filters_by_type(memory_store, memory_retriever):
    _save(memory_store, "用户偏好注释", type="user")
    _save(memory_store, "普通注释记录", type="fact")
    hits = memory_retriever.search("注释", types={"fact"}, limit=5)
    assert hits and all(h["type"] == "fact" for h in hits)
    assert len(hits) == 1


def test_search_filters_min_confidence(memory_store, memory_retriever):
    _save(memory_store, "低置信度内容", confidence=0.05)
    _save(memory_store, "高置信度内容", confidence=0.9)
    hits = memory_retriever.search("内容", min_conf=0.1, limit=5)
    assert hits and all(h["confidence"] >= 0.1 for h in hits)
    assert len(hits) == 1


def test_search_excludes_deprecated(memory_store, memory_retriever):
    mem_id = _save(memory_store, "已经废弃的内容")
    memory_store.update(mem_id, deprecated_by="decayed")
    hits = memory_retriever.search("废弃", limit=5)
    assert all(h["id"] != mem_id for h in hits)


def test_search_respects_limit_and_batch_fetch(memory_store, memory_retriever):
    # distinct docs that share the common token "条目" (kept apart from dedup)
    for doc in (
        "条目一：缓存设计要点",
        "条目二：日志轮转策略",
        "条目三：错误码规范说明",
        "条目四：部署流水线",
        "条目五：测试策略指南",
    ):
        _save(memory_store, doc)
    hits = memory_retriever.search("条目", limit=2)
    assert len(hits) == 2


def test_search_safe_with_fts_operator_terms(memory_store):
    """FTS5 operators (AND / OR / NOT / column:...) must be quoted, never parsed
    as syntax."""
    _save(memory_store, "AND 与 OR 逻辑说明")
    assert memory_store.bm25_search("AND")  # quoted literal, not operator
    for bad in ("OR", "not", "content:foo", "NEAR"):
        results = memory_store.bm25_search(bad)  # must not raise
        assert isinstance(results, list)
