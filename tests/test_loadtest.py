"""P2 压测: loadtest.aggregate (pure, no server needed)."""

import pytest

from eval_bench.loadtest import aggregate


def test_aggregate_success_rate_latency_tokens():
    results = [
        {"latency_ms": 100, "status": "success", "token_usage": 10, "ts_start": 0, "ts_end": 0.1},
        {"latency_ms": 200, "status": "success", "token_usage": 20, "ts_start": 0, "ts_end": 0.2},
        {"latency_ms": 300, "status": "failed", "token_usage": 5, "ts_start": 0, "ts_end": 0.3},
    ]
    s = aggregate(results)
    assert s["requests"] == 3
    assert s["success_rate"] == pytest.approx(2 / 3, abs=0.0001)
    assert s["p95_latency_ms"] == 200  # sorted [100,200,300], p95 index 1
    assert s["avg_latency_ms"] == pytest.approx(200)
    assert s["total_tokens"] == 35
    assert s["status_counts"]["success"] == 2
    assert s["qps"] > 0  # wall time 0.3s -> ~10 qps


def test_aggregate_empty():
    s = aggregate([])
    assert s["requests"] == 0 and s["success_rate"] == 0.0
