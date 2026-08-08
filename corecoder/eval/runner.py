"""Evaluation runner CLI (P1-1): benchmarks orchestration and writes a report.

Used by CI (continue-on-error, reports only) so the metrics feedback loop runs
without blocking merges.
"""

import argparse
import json
import sys
from dataclasses import asdict

from .metrics import OrchestrationMetrics, compute


def synthetic_traces(cases: int) -> list[dict]:
    """Generate deterministic-ish orchestration traces for the benchmark."""
    traces = []
    for i in range(cases):
        delegated_correctly = i % 5 != 0  # ~20% wrong delegation
        traces.append(
            {
                "delegation_correct": delegated_correctly,
                "serial_seconds": 12.0,
                "parallel_seconds": 4.0 + (i % 3),  # parallelism mostly pays
                "summary_tokens": 120,
                "raw_tokens": 1200,
                "lsp_calls": 3 if i % 2 == 0 else 0,
                "grep_calls": 4,
            }
        )
    return traces


def run_benchmark(cases: int = 20) -> dict:
    traces = synthetic_traces(cases)
    metrics: OrchestrationMetrics = compute(traces)
    return {
        "benchmark": "orchestration",
        "cases": cases,
        "metrics": asdict(metrics),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m corecoder.eval.runner")
    parser.add_argument("--benchmark", choices=["orchestration"], default="orchestration")
    parser.add_argument("--cases", type=int, default=20)
    parser.add_argument("--report", default=None, help="JSON report output path")
    args = parser.parse_args(argv)

    report = run_benchmark(args.cases)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
    metrics = report["metrics"]
    print(
        f"benchmark={report['benchmark']} cases={report['cases']} "
        f"delegation_accuracy={metrics['delegation_accuracy']:.2f} "
        f"speedup_ratio={metrics['speedup_ratio']:.2f} "
        f"context_inflation={metrics['context_inflation_ratio']:.2f} "
        f"lsp_adoption={metrics['lsp_adoption_rate']:.2f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
