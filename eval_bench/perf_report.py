"""Unified performance report — aggregate every perf data source into one snapshot.

Reads the JSON outputs of rag_eval / scorer / loadtest / judge_run (and accepts
optional in-process objects: LLMTracer, Agent._tool_metrics, compression_stats,
idempotency stats) and merges them into a single perf_report.json + Markdown, so
a full performance run ends in one answerable snapshot.

Offline: this only merges already-produced reports; it makes no LLM calls.

Usage:
    python -m eval_bench.perf_report \
        --rag results/perf/rag.json \
        --scorer results/<run>/summary.json \
        --loadtest results/loadtest-*/report.json \
        --judge results/perf/judge.json \
        --report results/perf/perf_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


def _r(value, nd=4) -> Any:
    """Round a float; pass through None."""
    try:
        return round(float(value), nd)
    except (TypeError, ValueError):
        return value


def build_report(
    *,
    rag: dict | None = None,
    scorer: dict | None = None,
    loadtest: dict | None = None,
    judge: dict | None = None,
    tracer=None,
    tool_metrics: dict | None = None,
    compression: dict | None = None,
    idem: dict | None = None,
) -> dict[str, Any]:
    """Merge every source into one report. Missing sources are simply absent —
    the report lists only what was measured."""
    report: dict[str, Any] = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}

    # --- latency / TTFT / tokens -------------------------------------------
    latency: dict[str, Any] = {}
    if tracer is not None:
        s = tracer.get_global_summary()
        latency.update(
            {
                "llm_calls": s["total_calls"],
                "llm_avg_ms": s["avg_duration_ms"],
                "llm_p95_ms": s["p95_duration_ms"],
                "ttft_avg_ms": s["avg_ttft_ms"],
                "ttft_p95_ms": s["p95_ttft_ms"],
            }
        )
    if loadtest:
        latency["task_avg_ms"] = _r(loadtest.get("avg_latency_ms"))
        latency["task_p95_ms"] = _r(loadtest.get("p95_latency_ms"))
    if latency:
        report["latency"] = latency

    # --- throughput (loadtest) ---------------------------------------------
    if loadtest:
        report["throughput"] = {
            "qps": _r(loadtest.get("qps")),
            "requests": loadtest.get("requests"),
            "success_rate": _r(loadtest.get("success_rate")),
            "status_counts": loadtest.get("status_counts"),
        }

    # --- cost / tokens / cache / compression --------------------------------
    cost: dict[str, Any] = {}
    if tracer is not None:
        s = tracer.get_global_summary()
        cost["total_tokens"] = s["total_tokens"]
    if loadtest and loadtest.get("total_tokens"):
        cost["loadtest_tokens"] = loadtest["total_tokens"]
    if idem:
        cost["cache_hit_rate"] = _r(idem.get("hit_rate"))
        cost["cache_hits"] = idem.get("hits")
        cost["cache_misses"] = idem.get("misses")
    if compression:
        cost["compression_ratio"] = _r(compression.get("avg_compression_ratio"))
        cost["tokens_saved"] = compression.get("tokens_saved")
        cost["compressions"] = compression.get("compressions")
    if cost:
        report["cost_tokens"] = cost

    # --- quality (scorer Pass@1 + tool metrics) -----------------------------
    quality: dict[str, Any] = {}
    if scorer:
        quality["pass_at_1"] = _r(scorer.get("pass_at_1"))
        quality["passed"] = scorer.get("passed")
        quality["total"] = scorer.get("total")
        quality["by_category"] = scorer.get("by_category")
    if tool_metrics:
        quality["tool_calls"] = tool_metrics.get("calls")
        quality["tool_success_rate"] = _r(tool_metrics.get("success_rate"))
        quality["tool_failure_rate"] = _r(tool_metrics.get("failure_rate"))
        quality["tool_retry_rate"] = _r(tool_metrics.get("retry_rate"))
        quality["tool_avg_ms"] = _r(tool_metrics.get("avg_duration_ms"))
        quality["tool_p95_ms"] = _r(tool_metrics.get("p95_duration_ms"))
    if quality:
        report["quality"] = quality

    # --- retrieval (rag_eval, plain-eval report) ----------------------------
    # A --compare report only has "rows"; a plain --report has recall@k etc.
    if rag and rag.get("recall@k") is not None:
        report["retrieval"] = {
            "recall_at_k": _r(rag.get("recall@k")),
            "precision_at_k": _r(rag.get("precision@k")),
            "citation_accuracy": _r(rag.get("citation_accuracy")),
        }
        ci = rag.get("citation_integrity")
        if isinstance(ci, dict):
            report["retrieval"]["citation_integrity"] = _r(ci.get("citation_integrity"))

    # --- generation (judge_run) ---------------------------------------------
    if judge:
        report["generation"] = {
            "judge_avg_score": _r(judge.get("avg_score")),
            "correct": judge.get("correct"),
            "partial": judge.get("partial"),
            "wrong": judge.get("wrong"),
        }

    # --- stability ----------------------------------------------------------
    stability: dict[str, Any] = {}
    if tracer is not None:
        s = tracer.get_global_summary()
        n = s["total_calls"]
        stability["llm_success_rate"] = _r((n - s["error_count"]) / n) if n else None
        stability["llm_errors"] = s["error_count"]
    if stability:
        report["stability"] = stability

    # --- attribution (rag --compare rows, if present) ------------------------
    rag_rows = rag.get("rows") if rag and isinstance(rag.get("rows"), dict) else None
    if rag_rows:
        report["attribution"] = {"rag": rag_rows}

    return report


def render_markdown(report: dict[str, Any]) -> str:
    """Human-readable performance report."""
    lines = ["# MyCoder Performance Report", "", f"- **Generated:** {report.get('generated_at')}", ""]

    def _kv(title: str, data: dict) -> None:
        if not data:
            return
        lines.append(f"## {title}")
        lines.append("")
        for k, v in data.items():
            lines.append(f"- **{k}:** {v}")
        lines.append("")

    for section in ("latency", "throughput", "cost_tokens", "quality", "retrieval", "generation", "stability"):
        _kv(section.replace("_", " ").title(), report.get(section))

    attr = report.get("attribution", {}).get("rag")
    if attr:
        lines.append("## RAG Optimization Attribution")
        lines.append("")
        lines.append("| pipeline | recall@k | precision@k |")
        lines.append("|---|---|---|")
        labels = {"pure_vector": "纯向量 (baseline)", "hybrid_rrf": "+ BM25 混合(RRF)", "hybrid_rerank": "+ Rerank"}
        for key, (rec, prec) in attr.items():
            label = labels.get(key, key)
            lines.append(f"| {label} | {rec:.1%} | {prec:.1%} |")
        lines.append("")
    return "\n".join(lines)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m eval_bench.perf_report", description=__doc__)
    p.add_argument("--rag", default=None, help="rag_eval report JSON")
    p.add_argument("--scorer", default=None, help="scorer summary.json")
    p.add_argument("--loadtest", default=None, help="loadtest report.json")
    p.add_argument("--judge", default=None, help="judge_run report JSON")
    p.add_argument("--report", default="results/perf/perf_report.json", help="output path (JSON)")
    args = p.parse_args(argv)

    report = build_report(
        rag=_load(Path(args.rag)) if args.rag else None,
        scorer=_load(Path(args.scorer)) if args.scorer else None,
        loadtest=_load(Path(args.loadtest)) if args.loadtest else None,
        judge=_load(Path(args.judge)) if args.judge else None,
    )
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md = out.with_suffix(".md")
    md.write_text(render_markdown(report), encoding="utf-8")
    print(f"[perf_report] -> {out}")
    print(f"[perf_report] -> {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
