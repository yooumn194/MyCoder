"""P1 Agentic RAG (memory/agentic.py) — retrieve, judge, refine or stop."""

from corecoder.memory import MemoryEntry, MemoryStore, HybridRetriever
from corecoder.memory.agentic import (
    AgenticRetriever,
    BroadeningRefiner,
    RuleSufficiencyEvaluator,
)


def _store(tmp_path):
    return MemoryStore(
        project_dir=tmp_path / "proj", global_dir=tmp_path / "glob", embedder=None
    )


def test_sufficient_first_round_stops(tmp_path):
    store = _store(tmp_path)
    store.save(MemoryEntry(content="缓存 策略 说明", confidence=0.8))
    r = AgenticRetriever(HybridRetriever(store))
    hits = r.search("缓存 策略", limit=5)
    assert any("缓存" in h["content"] for h in hits)


def test_refines_when_query_too_specific(tmp_path):
    store = _store(tmp_path)
    store.save(MemoryEntry(content="缓存 策略 说明", confidence=0.5))
    # 4-term query -> round 1 finds nothing -> refiner drops terms -> finds it
    r = AgenticRetriever(HybridRetriever(store), max_rounds=3)
    hits = r.search("缓存 策略 详细 说明", limit=5)
    assert any("缓存" in h["content"] for h in hits)


def test_max_rounds_capped_on_no_match(tmp_path):
    store = _store(tmp_path)
    store.save(MemoryEntry(content="完全不同的内容 说明"))
    r = AgenticRetriever(HybridRetriever(store), max_rounds=2)
    hits = r.search("不存在的词 也不存在", limit=5)
    assert hits == []  # best-effort empty after max rounds — no infinite loop


def test_rule_evaluator():
    e = RuleSufficiencyEvaluator(min_results=1, min_conf=0.3)
    assert e.is_sufficient("q", [{"confidence": 0.5}], 1) is True
    assert e.is_sufficient("q", [], 1) is False
    assert e.is_sufficient("q", [{"confidence": 0.1}], 1) is False


def test_broadening_refiner_drops_last_term():
    r = BroadeningRefiner()
    assert r.refine("缓存 策略 详细", [], 1) == "缓存 策略"
    assert r.refine("单一", [], 1) == "单一"


def test_agentic_skips_empty_query(tmp_path):
    store = _store(tmp_path)
    assert AgenticRetriever(HybridRetriever(store)).search("   ") == []
