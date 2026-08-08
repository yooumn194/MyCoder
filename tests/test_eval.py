"""Tests for the Phase 4 evaluation system (Module D)."""

from corecoder.eval import (
    FailureKnowledgeBase,
    FailurePattern,
    IncrementalDashboard,
    compute,
)
from corecoder.eval.metrics import (
    context_inflation_ratio,
    delegation_accuracy,
    lsp_adoption_rate,
    speedup_ratio,
)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def test_delegation_accuracy_metric():
    assert delegation_accuracy(correct=8, total=10) == 0.8
    assert delegation_accuracy(correct=0, total=0) == 1.0  # no delegations -> no error


def test_speedup_ratio_metric():
    assert speedup_ratio(serial_seconds=10, parallel_seconds=5) == 2.0
    assert speedup_ratio(serial_seconds=5, parallel_seconds=10) == 0.5  # parallelism hurt


def test_context_inflation_metric():
    assert context_inflation_ratio(summary_tokens=100, raw_tokens=1000) == 0.1
    assert context_inflation_ratio(summary_tokens=500, raw_tokens=0) == 0.0


def test_lsp_adoption_metric():
    assert lsp_adoption_rate(lsp_calls=8, grep_calls=2) == 0.8
    assert lsp_adoption_rate(lsp_calls=0, grep_calls=0) == 0.0


def test_compute_aggregates_traces():
    traces = [
        {"delegation_correct": True, "serial_seconds": 10, "parallel_seconds": 5,
         "summary_tokens": 100, "raw_tokens": 1000, "lsp_calls": 8, "grep_calls": 2},
        {"delegation_correct": False, "serial_seconds": 10, "parallel_seconds": 5,
         "summary_tokens": 100, "raw_tokens": 1000, "lsp_calls": 0, "grep_calls": 10},
    ]
    m = compute(traces)
    assert m.delegation_accuracy == 0.5
    assert m.speedup_ratio == 2.0
    assert m.context_inflation_ratio == 0.1
    assert m.lsp_adoption_rate == 0.4


# ---------------------------------------------------------------------------
# failure knowledge base
# ---------------------------------------------------------------------------

def test_failure_kb_record():
    kb = FailureKnowledgeBase()
    kb.record_failure("case-1", FailurePattern.TOOL_SELECTION, {"tool": "grep"})
    kb.record_failure("case-2", FailurePattern.TOOL_SELECTION, {"tool": "grep"})
    assert kb.get_trends()[FailurePattern.TOOL_SELECTION] == 2


def test_failure_kb_trend():
    kb = FailureKnowledgeBase()
    kb.record_failure("a", FailurePattern.DELEGATION, {})
    kb.record_failure("b", FailurePattern.CONTEXT_LOSS, {})
    trends = kb.get_trends()
    assert trends[FailurePattern.DELEGATION] == 1
    assert trends[FailurePattern.CONTEXT_LOSS] == 1


def test_failure_kb_suggest():
    kb = FailureKnowledgeBase()
    assert "工具描述" in kb.suggest_improvement(FailurePattern.TOOL_SELECTION)
    assert "Token 预算" in kb.suggest_improvement(FailurePattern.RESOURCE_EXHAUSTION)
    assert "摘要模板" in kb.suggest_improvement(FailurePattern.CONTEXT_LOSS)
    assert "委派决策" in kb.suggest_improvement(FailurePattern.DELEGATION)


# ---------------------------------------------------------------------------
# dashboard
# ---------------------------------------------------------------------------

def test_dashboard_add_result():
    d = IncrementalDashboard()
    assert d.add_result(passed=True)["status"] == "ok"
    assert d.add_result(passed=True)["pass_rate"] == 1.0


def test_dashboard_consecutive_failures_alarm():
    d = IncrementalDashboard()
    assert d.add_result(passed=False, pattern=FailurePattern.TOOL_SELECTION)["status"] == "ok"
    assert d.add_result(passed=False, pattern=FailurePattern.TOOL_SELECTION)["status"] == "ok"
    alarm = d.add_result(passed=False, pattern=FailurePattern.TOOL_SELECTION)
    assert alarm["status"] == "alarm"
    assert "TOOL_SELECTION" in alarm["message"]


def test_dashboard_success_resets_failures():
    d = IncrementalDashboard()
    d.add_result(passed=False)
    d.add_result(passed=False)
    assert d.add_result(passed=True)["status"] == "ok"
    assert d.consecutive_failures == 0


def test_dashboard_pass_rate():
    d = IncrementalDashboard()
    d.add_result(passed=True)
    d.add_result(passed=True)
    d.add_result(passed=False)
    assert d.add_result(passed=False)["pass_rate"] == 0.5
