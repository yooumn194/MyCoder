"""P1 wrong-memory pollution correction (audit_integrity + correct_memory + tool)."""

from corecoder.memory import MemoryEntry, MemoryStore
from corecoder.memory.maintenance import MemoryMaintainer
from corecoder.tools.memory_tools import MemoryCorrectTool


def _store(tmp_path):
    return MemoryStore(
        project_dir=tmp_path / "proj", global_dir=tmp_path / "glob", embedder=None
    )


def test_audit_flags_conflicting_memories(tmp_path):
    store = _store(tmp_path)
    a = store.save(MemoryEntry(content="JWT 令牌 有效期一小时", confidence=0.6, source="auto"))
    b = store.save(MemoryEntry(content="JWT 令牌 有效期两小时", confidence=0.6, source="auto"))
    c = store.save(MemoryEntry(content="缓存 策略 说明", confidence=0.6, source="auto"))

    issues = MemoryMaintainer(store).audit_integrity()
    conflicting = {i["id"] for i in issues if i["issue"] == "conflicting"}
    assert a in conflicting and b in conflicting  # same topic, different answers
    assert c not in conflicting  # unrelated topic untouched


def test_audit_flags_low_confidence(tmp_path):
    store = _store(tmp_path)
    store.save(MemoryEntry(content="不确定的记忆", confidence=0.1, source="auto"))
    issues = MemoryMaintainer(store).audit_integrity()
    assert any(i["issue"] == "low_confidence" for i in issues)


def test_correct_memory_replace_and_deprecate(tmp_path):
    store = _store(tmp_path)
    m = MemoryMaintainer(store)

    mid = store.save(MemoryEntry(content="错误的事实", confidence=0.5, source="auto"))
    assert m.correct_memory(mid, content="纠正后的事实")
    entry = store.get(mid)
    assert entry.content == "纠正后的事实"
    assert entry.deprecated_by is None
    assert entry.confidence >= 0.6

    mid2 = store.save(MemoryEntry(content="要废弃的错误"))
    assert m.correct_memory(mid2, reason="bad")
    assert store.get(mid2).deprecated_by == "bad"
    assert mid2 not in [e.id for e in store.list()]  # deprecated stops surfacing

    assert m.correct_memory("missing", content="x") is False


def test_memory_correct_tool(tmp_path):
    store = _store(tmp_path)
    mid = store.save(MemoryEntry(content="错误内容", confidence=0.5))

    out = MemoryCorrectTool(store=store).execute(memory_id=mid, content="正确内容")
    assert "纠正" in out
    assert store.get(mid).content == "正确内容"

    assert "未找到" in MemoryCorrectTool(store=store).execute(memory_id="nope")
