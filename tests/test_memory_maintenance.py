"""Phase 5 confidence decay + compaction (spec: 4 tests)."""

import time

from corecoder.memory import MemoryEntry
from corecoder.memory.maintenance import MemoryMaintainer


def _save(memory_store, content, **kw):
    defaults = dict(source="auto", confidence=0.8, access_count=5)
    defaults.update(kw)
    return memory_store.save(MemoryEntry(content=content, **defaults))


def test_decay_lowers_confidence_of_stale_auto_memory(memory_store):
    old = time.time() - 40 * 86400
    mem_id = _save(memory_store, "过期项目记忆", last_accessed=old)
    maintainer = MemoryMaintainer(memory_store, decay_days=30)
    assert maintainer.decay() == 1
    entry = memory_store.get(mem_id)
    # project-scoped factor 0.8
    assert abs(entry.confidence - 0.8 * 0.8) < 1e-6


def test_decay_skips_fresh_and_confirmed(memory_store):
    _save(memory_store, "新记忆不应衰减", last_accessed=time.time())
    confirmed = _save(
        memory_store,
        "已确认记忆",
        source="confirmed",
        confidence=1.0,
        last_accessed=time.time() - 40 * 86400,
    )
    maintainer = MemoryMaintainer(memory_store, decay_days=30)
    assert maintainer.decay() == 0
    assert memory_store.get(confirmed).confidence == 1.0  # never decays


def test_global_decays_slower_than_project(memory_store):
    old = time.time() - 40 * 86400
    pid = _save(memory_store, "项目记忆", last_accessed=old, scope="project")
    gid = _save(memory_store, "全局记忆", last_accessed=old, scope="global")
    MemoryMaintainer(memory_store, decay_days=30).decay()
    p_conf = memory_store.get(pid).confidence
    g_conf = memory_store.get(gid).confidence
    assert abs(p_conf - 0.8 * 0.8) < 1e-6
    assert abs(g_conf - 0.8 * 0.95) < 1e-6
    assert g_conf > p_conf  # global memories decay slower


def test_decay_marks_below_threshold_and_compact_prunes(memory_store):
    old = time.time() - 40 * 86400
    doomed = _save(memory_store, "低频访问将废弃", confidence=0.14, last_accessed=old)
    keeper = _save(memory_store, "正常保留", confidence=0.5, last_accessed=time.time())

    maintainer = MemoryMaintainer(memory_store, decay_days=30, confidence_threshold=0.15)
    assert maintainer.decay() == 1
    # 0.14 * 0.8 = 0.112 < 0.15 -> flagged deprecated_by='decayed'
    assert memory_store.get(doomed).deprecated_by == "decayed"

    removed = maintainer.compact()
    assert removed == 1
    assert memory_store.get(doomed) is None
    assert memory_store.get(keeper) is not None


def test_get_stats_reports_counts_and_distribution(memory_store):
    _save(memory_store, "第一条", type="fact")
    _save(memory_store, "第二条", type="project", scope="global")
    stats = MemoryMaintainer(memory_store).get_stats()
    assert stats["total"] == 2
    assert stats["active"] == 2
    assert stats["by_type"].get("fact") == 1
    assert stats["by_scope"].get("global") == 1
    assert 0.0 <= stats["avg_confidence"] <= 1.0
