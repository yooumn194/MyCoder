"""P0 multi-turn query rewriting (memory/query_rewrite.py + retriever hook)."""

from mycoder.memory import MemoryEntry, MemoryStore, HybridRetriever
from mycoder.memory.query_rewrite import (
    IdentityQueryRewriter,
    LLMQueryRewriter,
)


class _Resp:
    def __init__(self, content):
        self.content = content


class _Stub:
    def __init__(self, content):
        self._content = content

    def chat(self, messages, response_format=None):
        return _Resp(self._content)


def test_identity_rewriter_is_noop():
    r = IdentityQueryRewriter()
    assert r.rewrite("认证模块", history=[{"role": "user", "content": "上轮"}] * 3) == "认证模块"


def test_llm_rewriter_rewrites_with_history():
    stub = _Stub('{"query": "JWT 令牌过期策略"}')
    r = LLMQueryRewriter(llm=stub)
    out = r.rewrite("那它呢？", history=[{"role": "user", "content": "讲一下 JWT 令牌"}])
    assert out == "JWT 令牌过期策略"


def test_llm_rewriter_falls_back_on_bad_output():
    r = LLMQueryRewriter(llm=_Stub("不是 JSON 的一堆话"))
    assert r.rewrite("那它呢？", history=[{"role": "user", "content": "x"}]) == "那它呢？"


def test_llm_rewriter_without_llm_returns_query():
    r = LLMQueryRewriter(llm=None)
    assert r.rewrite("原始问题", history=None) == "原始问题"


def test_retriever_applies_rewriter_when_history_given(tmp_path):
    store = MemoryStore(
        project_dir=tmp_path / "p", global_dir=tmp_path / "g", embedder=None
    )
    store.save(MemoryEntry(content="JWT 令牌过期策略说明", type="fact"))

    class _FixedRewriter:
        def rewrite(self, query, history=None):
            return "JWT 令牌过期策略"

    retriever = HybridRetriever(store, query_rewriter=_FixedRewriter())

    # with history -> rewritten query hits the doc
    with_history = retriever.search("那它呢？", limit=5, history=[{"role": "user", "content": "x"}])
    assert any("JWT 令牌" in h["content"] for h in with_history)

    # without history -> raw query used, no match
    without = retriever.search("那它呢？", limit=5)
    assert not any("JWT 令牌" in h["content"] for h in without)


def test_llm_rewriter_uses_messages_with_history(tmp_path):
    seen = {}

    class _CapturingStub:
        def chat(self, messages, response_format=None):
            seen["user"] = messages[1]["content"]
            return _Resp('{"query": "缓存策略"}')

    r = LLMQueryRewriter(llm=_CapturingStub())
    r.rewrite("那它？", history=[{"role": "user", "content": "讲一下缓存"}])
    assert "讲一下缓存" in seen["user"]  # history is passed into the prompt
