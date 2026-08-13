"""P2 评测定量升级: runner variant + scorer --compare delta report."""

import pytest

from eval_bench.runner import _result
from eval_bench.scorer import compute_comparison, render_comparison_markdown

_IDS = ["b1", "b2", "r1"]


def _results(passed: set[str]) -> list[dict]:
    return [
        {
            "id": i,
            "category": "bugfix" if i.startswith("b") else "refactor",
            "difficulty": "easy",
            "agent_status": "success" if i in passed else "failed",
            "tests_passed": 1 if i in passed else 0,
            "tests_total": 1,
            "duration_s": 1.0,
            "token_usage": 100,
            "error_class": None if i in passed else "SUBAGENT_TIMEOUT",
            "error_msg": None,
            "variant": "default",
        }
        for i in _IDS
    ]


def test_result_record_carries_variant():
    r = _result(
        {"id": "x", "category": "bugfix", "difficulty": "easy"},
        "success", 1, 1, 1.0, 10, None, None, variant="agentic",
    )
    assert r["variant"] == "agentic"


def test_compute_comparison_delta():
    base = _results({"b1"})                # 1/3 pass
    treat = _results({"b1", "b2", "r1"})   # 3/3 pass
    cmp = compute_comparison(base, treat)
    # 1/3 -> 3/3, so Δ = 2/3 (rounded to 4 decimals by the scorer)
    assert cmp["delta_pass_at_1"] == pytest.approx(1.0 - 1 / 3, abs=0.0001)
    assert cmp["by_category"]["bugfix"]["delta"] == pytest.approx(0.5)  # 1/2 -> 2/2
    assert cmp["by_difficulty"]["easy"]["treatment_pass_rate"] == 1.0
    assert cmp["delta_avg_tokens"] == 0  # token usage unchanged


def test_render_comparison_markdown():
    cmp = compute_comparison(_results({"b1"}), _results({"b1", "b2", "r1"}))
    md = render_comparison_markdown(cmp, "baseline", "agentic")
    assert "Δ Pass@1" in md
    assert "bugfix" in md and "+" in md  # positive delta rendered


def test_perf_stats_aggregation():
    from eval_bench.scorer import _perf_stats

    results = [
        {"id": "a", "perf": {"llm_calls": 5, "prompt_tokens": 100, "completion_tokens": 50,
                              "total_tokens": 150, "avg_latency_ms": 100, "p95_latency_ms": 200,
                              "error_calls": 0, "cost_usd": 0.01}},
        {"id": "b", "perf": {"llm_calls": 10, "prompt_tokens": 200, "completion_tokens": 100,
                              "total_tokens": 300, "avg_latency_ms": 200, "p95_latency_ms": 400,
                              "error_calls": 1, "cost_usd": 0.02}},
        {"id": "c", "agent_status": "failed", "error_class": "X"},  # no perf captured
    ]
    s = _perf_stats(results)
    assert s["tasks_with_perf"] == 2
    assert s["llm_calls"] == 15
    assert s["total_tokens"] == 450
    assert s["avg_latency_ms"] == pytest.approx(150)
    assert s["p95_latency_ms"] == 200  # sorted [200,400], small-sample p95 index 0
    assert s["cost_usd"] == pytest.approx(0.03)
    assert s["avg_tokens_per_task"] == pytest.approx(225)


def test_result_includes_perf():
    r = _result(
        {"id": "x", "category": "bugfix", "difficulty": "easy"},
        "success", 1, 1, 1.0, 10, None, None, variant="v", perf={"llm_calls": 5},
    )
    assert r["perf"] == {"llm_calls": 5}
    assert r["variant"] == "v"
