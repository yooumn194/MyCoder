"""Load test for the CoreCoder HTTP API (P2, 压测).

Drives POST /v1/agent/run under concurrency, polls /status to a terminal state,
and aggregates QPS / p95 latency / success rate / token usage.

Requires a running server with an LLM key (each request executes a real agent
run — this is a real load test, not a smoke test). The `aggregate` function is
pure and unit-testable without a server.

Usage:
    python -m eval_bench.loadtest --base-url http://localhost:8000 \
        --concurrency 4 --requests 20
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore[assignment]

_POLL_INTERVAL = 0.5
_TERMINAL = {"success", "failed"}


def aggregate(results: list[dict]) -> dict:
    """Per-request latency / status / tokens -> SLO-style summary."""
    n = len(results)
    empty = {
        "requests": 0,
        "success_rate": 0.0,
        "p95_latency_ms": 0.0,
        "avg_latency_ms": 0.0,
        "total_tokens": 0,
        "qps": 0.0,
        "status_counts": {},
    }
    if n == 0:
        return empty
    latencies = sorted(r["latency_ms"] for r in results)
    success = sum(1 for r in results if r["status"] == "success")
    p95 = latencies[min(n - 1, int(n * 0.95) - 1)]
    avg = sum(latencies) / n
    total_tokens = sum(int(r.get("token_usage") or 0) for r in results)
    wall = max(r.get("ts_end", 0) for r in results) - min(
        r.get("ts_start", 0) for r in results
    )
    return {
        "requests": n,
        "success_rate": round(success / n, 4),
        "p95_latency_ms": round(p95, 1),
        "avg_latency_ms": round(avg, 1),
        "total_tokens": total_tokens,
        "qps": round(n / max(wall, 0.001), 2),
        "status_counts": dict(Counter(r.get("status", "unknown") for r in results)),
    }


def run_request(client, base_url: str, session_id: str, task: str, timeout_s: float) -> dict:
    """POST /run, poll /status to terminal, return one result record."""
    ts_start = time.monotonic()
    try:
        resp = client.post(
            f"{base_url}/v1/agent/run",
            json={"task": task, "session_id": session_id},
            timeout=30,
        )
        if resp.status_code != 202:
            return {
                "latency_ms": round((time.monotonic() - ts_start) * 1000, 1),
                "status": "rejected",
                "token_usage": None,
                "ts_start": ts_start,
                "ts_end": time.monotonic(),
            }
        deadline = ts_start + timeout_s
        status, token_usage = "timeout", None
        while time.monotonic() < deadline:
            sr = client.get(f"{base_url}/v1/agent/status/{session_id}", timeout=30)
            if sr.status_code == 200:
                data = sr.json()
                status = data.get("status", status)
                token_usage = data.get("token_usage")
                if status in _TERMINAL:
                    break
            time.sleep(_POLL_INTERVAL)
    except Exception as exc:  # noqa: BLE001 - a flaky request must not kill the run
        status, token_usage = f"error:{type(exc).__name__}", None
    return {
        "latency_ms": round((time.monotonic() - ts_start) * 1000, 1),
        "status": status,
        "token_usage": token_usage,
        "ts_start": ts_start,
        "ts_end": time.monotonic(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval_bench.loadtest", description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--task", default="写一个返回两数之和的 python 函数", help="task sent to /run")
    parser.add_argument("--timeout", type=float, default=120.0, help="per-request watchdog (s)")
    parser.add_argument("--session-prefix", default="lt")
    parser.add_argument("--results", default=None, help="output dir (default results/loadtest-<ts>)")
    args = parser.parse_args(argv)

    if httpx is None:
        print("[loadtest] httpx is required")
        return 1
    results_dir = Path(args.results) if args.results else (
        Path("results") / f"loadtest-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex[:8]

    client = httpx.Client()
    results: list[dict] = []
    print(f"[loadtest] {args.requests} requests, concurrency={args.concurrency}, base={args.base_url}")
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futures = [
            pool.submit(
                run_request, client, args.base_url, f"{args.session_prefix}-{run_id}-{i}",
                args.task, args.timeout,
            )
            for i in range(args.requests)
        ]
        for fut in as_completed(futures):
            results.append(fut.result())
    client.close()

    summary = aggregate(results)
    (results_dir / "report.json").write_text(
        __import__("json").dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[loadtest] requests={summary['requests']} success_rate={summary['success_rate']:.1%} "
        f"p95={summary['p95_latency_ms']}ms avg={summary['avg_latency_ms']}ms "
        f"qps={summary['qps']} tokens={summary['total_tokens']}"
    )
    print(f"[loadtest] report -> {results_dir}/report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
