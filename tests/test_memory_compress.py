"""P1 memory compression — conversation → long-term memory closure.

Covers memory/compressor.py + its wiring: structured fact extraction into the
memory DB, LLM-judge retention audit, memory-DB rolling update (demote, not
delete), and the ContextManager.on_compressed hook.
"""

from unittest import mock

from mycoder.context import ContextManager
from mycoder.memory.compressor import MemoryCompressor
from mycoder.memory.store import MemoryStore
from mycoder.memory.types import MemoryEntry


def _store(tmp_path) -> MemoryStore:
    return MemoryStore(
        project_dir=tmp_path / "proj", global_dir=tmp_path / "glob", embedder=None
    )


class _R:
    pass


def _stub(content):
    """A fake LLM that always returns the same content."""

    class _S:
        def chat(self, messages, response_format=None):
            r = _R()
            r.content = content
            return r

    return _S()


def _seq_stub(contents):
    """A fake LLM that returns one content per call, in order."""

    class _S:
        def __init__(self):
            self._contents = list(contents)

        def chat(self, messages, response_format=None):
            r = _R()
            r.content = self._contents.pop(0)
            return r

    return _S()


# ------------------------------------------------------- Phase 3 extraction
def test_extract_facts_rule_fallback(tmp_path):
    """No LLM -> zero-dependency rule extraction surfaces files + errors."""
    c = MemoryCompressor(store=_store(tmp_path), llm=None)
    msgs = [
        {"role": "user", "content": "修复 mycoder/agent.py 的 bug"},
        {"role": "tool", "content": "error: name 'x' is not defined"},
    ]
    facts = c.extract_facts(msgs)
    assert any("mycoder/agent.py" in f["fact"] for f in facts)
    assert any("error" in f["fact"].lower() for f in facts)


def test_extract_facts_llm_structured(tmp_path):
    c = MemoryCompressor(
        store=_store(tmp_path),
        llm=_stub(
            '{"facts": [{"fact": "项目用 FastAPI", "type": "project", "confidence": 0.9},'
            '{"fact": "放弃重构", "type": "decision", "confidence": 0.7}]}'
        ),
    )
    facts = c.extract_facts([{"role": "user", "content": "我们决定用 FastAPI"}])
    assert len(facts) == 2
    assert facts[0]["type"] == "project"
    assert facts[0]["confidence"] == 0.9
    assert facts[1]["type"] == "decision"


def test_extract_facts_llm_failure_falls_back(tmp_path):
    """Malformed judge output / exceptions fall back to rules, never raise."""
    c = MemoryCompressor(store=_store(tmp_path), llm=_stub("not json at all"))
    facts = c.extract_facts([{"role": "user", "content": "error in a.py"}])
    assert isinstance(facts, list)  # rule fallback, not a crash


def test_settle_facts_persists_to_store(tmp_path):
    store = _store(tmp_path)
    c = MemoryCompressor(
        store=store,
        llm=_stub('{"facts": [{"fact": "约定：提交信息用英文", "type": "feedback", "confidence": 0.8}]}'),
    )
    ids = c.settle_facts([{"role": "user", "content": "记住提交信息用英文"}])
    assert len(ids) == 1
    entry = store.get(ids[0])
    assert entry is not None
    assert "提交信息用英文" in entry.content
    assert entry.type == "feedback"


# ------------------------------------------------------------- loss audit
def test_verify_retention_with_and_without_llm(tmp_path):
    c = MemoryCompressor(
        store=_store(tmp_path), llm=_stub('{"retention": 0.8, "lost": "细节"}')
    )
    assert c.verify_retention("原文", "摘要") == 0.8

    c2 = MemoryCompressor(store=_store(tmp_path), llm=None)
    assert c2.verify_retention("原文", "摘要") is None


# ---------------------------------------------------- compress hook closure
def test_on_compressed_demotes_to_memory(tmp_path):
    store = _store(tmp_path)
    c = MemoryCompressor(
        store=store,
        llm=_seq_stub(
            [
                '{"facts": [{"fact": "决策：用向量库", "type": "decision", "confidence": 0.8}]}',
                '{"retention": 0.9, "lost": "无"}',
            ]
        ),
    )
    out = c.on_compressed([{"role": "user", "content": "决定用向量库"}], "摘要")
    assert out["facts_saved"] == 1
    assert out["retention"] == 0.9
    entries = store.list(include_deprecated=False)
    assert any("向量库" in e.content for e in entries)


def test_on_compressed_never_raises(tmp_path):
    c = MemoryCompressor(store=_store(tmp_path), llm=None)
    assert c.on_compressed([{"role": "user", "content": "x"}], "s") == {
        "facts_saved": 0,
        "retention": None,
    }


# ------------------------------------------------- rolling update (demote)
def test_summarize_cluster_demotes_not_deletes(tmp_path):
    store = _store(tmp_path)
    for i in range(6):
        store.save(
            MemoryEntry(
                content=f"低价值记忆{i}",
                type="fact",
                scope="project",
                source="auto",
                confidence=0.4,
            )
        )
    c = MemoryCompressor(
        store=store,
        llm=_stub('{"summary": "合并摘要：六条低价值记忆"}'),
    )
    n = c.summarize_cluster(scope="project", min_count=5)
    assert n == 6

    # a summary entry now exists
    active = store.list(scope="project", include_deprecated=False)
    assert any("合并摘要" in e.content for e in active)

    # originals are DEMOTED (deprecated_by='merged:...'), not deleted
    all_entries = store.list(scope="project", include_deprecated=True)
    deprecated = [e for e in all_entries if e.deprecated_by]
    assert len(deprecated) == 6
    assert all(d.deprecated_by.startswith("merged:") for d in deprecated)


def test_summarize_cluster_needs_min_count(tmp_path):
    store = _store(tmp_path)
    store.save(
        MemoryEntry(content="只有一条", type="fact", scope="project", source="auto", confidence=0.4)
    )
    c = MemoryCompressor(store=store, llm=_stub('{"summary": "x"}'))
    assert c.summarize_cluster(scope="project", min_count=5) == 0


def test_summarize_cluster_without_llm_is_noop(tmp_path):
    store = _store(tmp_path)
    c = MemoryCompressor(store=store, llm=None)
    assert c.summarize_cluster(scope="project", min_count=1) == 0


# ------------------------------------------------------------- context hook
def test_context_compress_fires_hook():
    fired = []
    ctx = ContextManager(
        max_tokens=2000, on_compressed=lambda old, summary: fired.append(len(old))
    )
    msgs = [{"role": "user", "content": f"m{i} " + "a" * 300} for i in range(20)]
    ctx.maybe_compress(msgs, None)
    assert fired  # a summarization pass fired the hook
    assert fired[0] > 0  # the compressed old turns were handed over


def test_context_no_hook_by_default():
    assert ContextManager(max_tokens=2000).on_compressed is None


# ------------------------------------------------------------- agent wiring
def test_agent_wires_memory_compressor():
    from mycoder.agent import Agent
    from mycoder.llm import LLM

    compressor = mock.Mock()
    agent = Agent(llm=LLM.__new__(LLM), memory_compressor=compressor)
    assert agent.context.on_compressed == compressor.on_compressed
