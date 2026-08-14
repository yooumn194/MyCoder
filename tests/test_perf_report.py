"""Unified perf-report aggregation (eval_bench/perf_report.py)."""

from eval_bench.perf_report import build_report, render_markdown


def test_perf_report_aggregates_all_sources():
    report = build_report(
        rag={"recall@k": 1.0, "precision@k": 0.8, "citation_accuracy": 0.9},
        scorer={"pass_at_1": 0.7, "passed": 21, "total": 30},
        loadtest={"qps": 10.0, "requests": 20, "success_rate": 0.85, "p95_latency_ms": 4000.0},
        judge={"avg_score": 0.8, "correct": 3, "partial": 1, "wrong": 0},
        idem={"hit_rate": 0.4, "hits": 2, "misses": 3},
        compression={"avg_compression_ratio": 0.5, "tokens_saved": 1000, "compressions": 2},
    )
    assert report["retrieval"]["recall_at_k"] == 1.0
    assert report["quality"]["pass_at_1"] == 0.7
    assert report["throughput"]["qps"] == 10.0
    assert report["generation"]["judge_avg_score"] == 0.8
    assert report["cost_tokens"]["cache_hit_rate"] == 0.4
    assert report["cost_tokens"]["compression_ratio"] == 0.5


def test_compare_report_has_attribution_no_null_retrieval():
    """A rag --compare report only has rows — no null recall@k section."""
    report = build_report(
        rag={"rows": {"pure_vector": [0.8, 0.6], "hybrid_rrf": [1.0, 0.8]}}
    )
    assert "retrieval" not in report
    assert report["attribution"]["rag"]["hybrid_rrf"] == [1.0, 0.8]


def test_perf_report_markdown_renders():
    report = build_report(loadtest={"qps": 5.0, "requests": 10, "success_rate": 1.0})
    md = render_markdown(report)
    assert "# MyCoder Performance Report" in md
    assert "**qps:** 5.0" in md
