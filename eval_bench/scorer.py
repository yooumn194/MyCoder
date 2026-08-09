"""Eval scorer — Pass@1 statistics + human-readable report.

Reads results/<run>/raw_results.json (produced by runner.py) and writes:
  * summary.json      — machine-readable stats
  * report.md         — human-readable report
  * chart.png         — optional category bar chart (needs matplotlib)

Pass@1 definition: a problem passes when agent_status == 'success' AND its
verification tests all passed (error_class is None). Failure reasons are
classified from the error_class recorded by the runner (real envelope error
codes like CIRCUIT_BREAKER_OPEN / SUBAGENT_TIMEOUT / TOKEN_BUDGET_EXCEEDED,
plus runner-side classes TIMEOUT / VERIFICATION_FAILED / AGENT_FAILED).

Usage:
    python -m eval_bench.scorer --results results/run-...
    python -m eval_bench.scorer --results results/run-... --chart
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def load_results(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def is_pass(r: dict) -> bool:
    return r.get("agent_status") == "success" and r.get("error_class") is None


def group_pass_rate(results: list[dict], key: str) -> dict[str, dict]:
    buckets: dict[str, dict] = {}
    for r in results:
        bucket = buckets.setdefault(str(r.get(key, "?")), {"pass": 0, "total": 0})
        bucket["total"] += 1
        bucket["pass"] += int(is_pass(r))
    return {k: {**v, "pass_rate": round(v["pass"] / v["total"], 4) if v["total"] else 0.0} for k, v in sorted(buckets.items())}


def failure_distribution(results: list[dict]) -> dict[str, int]:
    dist = Counter(
        r.get("error_class") or ("PASS" if is_pass(r) else "UNKNOWN")
        for r in results
    )
    return dict(sorted(dist.items(), key=lambda kv: (-kv[1], kv[0])))


def average(result_list: list[dict], field: str) -> float:
    vals = [r.get(field) for r in result_list if r.get(field) is not None]
    return round(sum(vals) / len(vals), 2) if vals else 0.0


def compute_stats(results: list[dict]) -> dict:
    passed = [r for r in results if is_pass(r)]
    return {
        "total": len(results),
        "passed": len(passed),
        "pass_at_1": round(len(passed) / len(results), 4) if results else 0.0,
        "by_category": group_pass_rate(results, "category"),
        "by_difficulty": group_pass_rate(results, "difficulty"),
        "failure_distribution": failure_distribution(results),
        "avg_duration_s": average(results, "duration_s"),
        "avg_tokens": average(results, "token_usage"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# --------------------------------------------------------------------------- report
def _table(header: list[str], rows: list[list]) -> str:
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(header)]
    line = "| " + " | ".join(str(h).ljust(w) for h, w in zip(header, widths)) + " |"
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    body = ["| " + " | ".join(str(c).ljust(w) for c, w in zip(row, widths)) + " |" for row in rows]
    return "\n".join([line, sep, *body])


def render_markdown(stats: dict, results: list[dict], env: dict) -> str:
    lines: list[str] = []
    lines.append("# CoreCoder Eval Report")
    lines.append("")
    lines.append(f"- **Date:** {stats['generated_at']}")
    lines.append(f"- **Environment:** Python {env['python']}, OS {env['os']}, base_url {env['base_url']}")
    lines.append(f"- **Dataset:** {env['dataset']}")
    lines.append(f"- **Total Pass@1:** **{stats['pass_at_1'] * 100:.1f}%** ({stats['passed']}/{stats['total']})")
    lines.append(f"- **Avg duration:** {stats['avg_duration_s']}s  ·  **Avg tokens:** {stats['avg_tokens']}")
    lines.append("")

    lines.append("## Pass@1 by category")
    lines.append(_table(
        ["category", "pass", "total", "pass_rate"],
        [[k, v["pass"], v["total"], f"{v['pass_rate'] * 100:.1f}%"] for k, v in stats["by_category"].items()],
    ))
    lines.append("")

    lines.append("## Pass@1 by difficulty")
    lines.append(_table(
        ["difficulty", "pass", "total", "pass_rate"],
        [[k, v["pass"], v["total"], f"{v['pass_rate'] * 100:.1f}%"] for k, v in stats["by_difficulty"].items()],
    ))
    lines.append("")

    lines.append("## Failure distribution")
    lines.append(_table(
        ["error_class", "count"],
        [[k, str(v)] for k, v in stats["failure_distribution"].items()],
    ))
    lines.append("")

    failed = [r for r in results if not is_pass(r)]
    lines.append("## Failed cases")
    if failed:
        lines.append(_table(
            ["id", "category", "agent_status", "tests", "error_class", "detail"],
            [[r["id"], r["category"], r.get("agent_status"), f"{r.get('tests_passed')}/{r.get('tests_total')}", r.get("error_class"), str(r.get("error_msg") or "")[:40]] for r in failed],
        ))
    else:
        lines.append("_None — all problems passed._")
    lines.append("")
    return "\n".join(lines)


def render_rich(stats: dict) -> None:
    """Terminal table via rich (guarded — pure-stdlib fallback prints plain)."""
    try:
        from rich.console import Console
        from rich.table import Table
    except ImportError:
        print(f"Pass@1: {stats['pass_at_1'] * 100:.1f}% ({stats['passed']}/{stats['total']})")
        return
    console = Console()
    table = Table(title=f"CoreCoder Eval — Pass@1 {stats['pass_at_1'] * 100:.1f}%")
    for col in ("dimension", "value", "pass", "total", "rate"):
        table.add_column(col)
    for label, group in (("category", stats["by_category"]), ("difficulty", stats["by_difficulty"])):
        for k, v in group.items():
            table.add_row(label, k, str(v["pass"]), str(v["total"]), f"{v['pass_rate'] * 100:.1f}%")
    table.add_row("overall", "", str(stats["passed"]), str(stats["total"]), f"{stats['pass_at_1'] * 100:.1f}%")
    console.print(table)
    print(f"Avg duration: {stats['avg_duration_s']}s  ·  Avg tokens: {stats['avg_tokens']}")


def render_chart(stats: dict, out: Path) -> bool:
    """Optional matplotlib bar chart; returns False when matplotlib is missing."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    cats = list(stats["by_category"])
    rates = [stats["by_category"][c]["pass_rate"] * 100 for c in cats]
    plt.figure(figsize=(7, 4))
    bars = plt.bar(cats, rates, color="#4c72b0")
    for bar, rate in zip(bars, rates):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{rate:.0f}%", ha="center", va="bottom")
    plt.ylim(0, 105)
    plt.ylabel("Pass@1 (%)")
    plt.title("CoreCoder Eval — Pass@1 by category")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval_bench.scorer", description=__doc__)
    parser.add_argument("--results", required=True, help="results/<run> directory containing raw_results.json")
    parser.add_argument("--chart", action="store_true", help="also render chart.png (needs matplotlib)")
    parser.add_argument("--base-url", default="", help="API base URL recorded in the report")
    args = parser.parse_args(argv)

    results_dir = Path(args.results)
    raw = results_dir / "raw_results.json"
    if not raw.exists():
        print(f"[scorer] no raw_results.json in {results_dir}")
        return 1
    results = load_results(raw)
    stats = compute_stats(results)

    env = {
        "python": platform.python_version(),
        "os": platform.system(),
        "base_url": args.base_url or "(not recorded)",
        "dataset": "eval_bench/dataset.json",
    }
    (results_dir / "summary.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    (results_dir / "report.md").write_text(render_markdown(stats, results, env), encoding="utf-8")

    if args.chart:
        chart = results_dir / "chart.png"
        if render_chart(stats, chart):
            print(f"[scorer] chart -> {chart}")
        else:
            print("[scorer] matplotlib not installed — skipped chart")

    render_rich(stats)
    print(f"[scorer] summary.json + report.md written to {results_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
