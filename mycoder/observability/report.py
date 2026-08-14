"""Monitor report — one snapshot across every observability source (P3).

Aggregates the LLM trace (global + per-session: latency / tokens / TTFT /
errors / cost), production run success rate (from a StateBackend), into a
single answerable dict — the "监控报告" that turns scattered metrics into one
story. Served by GET /v1/agent/report.
"""

from __future__ import annotations

import time
from typing import Any


def build_monitor_report(
    tracer,
    *,
    sessions: list[dict] | None = None,
    price_per_1k: dict | None = None,
) -> dict[str, Any]:
    """Aggregate one monitor snapshot.

    tracer: LLMTracer (global + per-session LLM metrics).
    sessions: list of {status, ...} run records from a StateBackend (production
        success rate); None skips that section.
    price_per_1k: price table for cost; None -> costs skipped.
    """
    global_llm = tracer.get_global_summary()
    session_ids = tracer.list_sessions()
    per_session: dict[str, dict[str, Any]] = {}
    total_cost = 0.0
    for sid in session_ids:
        s = tracer.get_session_summary(sid)
        cost = 0.0
        if price_per_1k:
            try:
                cost = float(
                    tracer.get_cost_estimate(sid, price_per_1k=price_per_1k)[
                        "total_cost_usd"
                    ]
                )
            except Exception:  # noqa: BLE001 - cost is best-effort
                cost = 0.0
        total_cost += cost
        per_session[sid] = {
            "calls": s["total_calls"],
            "tokens": s["total_tokens"],
            "avg_duration_ms": s["avg_duration_ms"],
            "p95_duration_ms": s["p95_duration_ms"],
            "avg_ttft_ms": s["avg_ttft_ms"],
            "errors": s["error_count"],
            "cost_usd": round(cost, 6),
        }

    total_calls = global_llm["total_calls"]
    report: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "llm": {
            "calls": total_calls,
            "tokens": global_llm["total_tokens"],
            "avg_duration_ms": global_llm["avg_duration_ms"],
            "p95_duration_ms": global_llm["p95_duration_ms"],
            "avg_ttft_ms": global_llm["avg_ttft_ms"],
            "p95_ttft_ms": global_llm["p95_ttft_ms"],
            "errors": global_llm["error_count"],
            "success_rate": round(
                (total_calls - global_llm["error_count"]) / total_calls, 4
            )
            if total_calls
            else 0.0,
            "cost_usd": round(total_cost, 4),
            "sessions": len(session_ids),
        },
        "per_session": per_session,
    }

    if sessions:
        done = [s for s in sessions if s.get("status") in ("success", "failed")]
        success = sum(1 for s in done if s.get("status") == "success")
        report["production_runs"] = {
            "total": len(sessions),
            "completed": len(done),
            "running": len(sessions) - len(done),
            "success": success,
            "failed": len(done) - success,
            "success_rate": round(success / len(done), 4) if done else 0.0,
        }
    return report
