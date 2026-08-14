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


def badcases(results: list[dict]) -> list[dict]:
    """Failed results extracted as a badcase 回流 list — each record carries
    enough context (id/category/error_class/error_msg/perf) to feed back into
    the dataset or an SFT pass."""
    return [
        {
            "id": r.get("id"),
            "category": r.get("category"),
            "difficulty": r.get("difficulty"),
            "agent_status": r.get("agent_status"),
            "tests": f"{r.get('tests_passed')}/{r.get('tests_total')}",
            "error_class": r.get("error_class"),
            "error_msg": r.get("error_msg"),
            "duration_s": r.get("duration_s"),
            "variant": r.get("variant"),
            "perf": r.get("perf"),
        }
        for r in results
        if not is_pass(r)
    ]


def write_badcases(results: list[dict], path: Path) -> int:
    """Write the badcase 回流 list to `path`; returns the number of bad cases."""
    cases = badcases(results)
    payload = {
        "total_badcases": len(cases),
        "badcases": cases,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(cases)


def failure_distribution(results: list[dict]) -> dict[str, int]:
    dist = Counter(
        r.get("error_class") or ("PASS" if is_pass(r) else "UNKNOWN")
        for r in results
    )
    return dict(sorted(dist.items(), key=lambda kv: (-kv[1], kv[0])))


def average(result_list: list[dict], field: str) -> float:
    vals = [r.get(field) for r in result_list if r.get(field) is not None]
    return round(sum(vals) / len(vals), 2) if vals else 0.0


def _perf_stats(results: list[dict]) -> dict:
    """Aggregate per-task `perf` dicts (worker's LLM-trace summary) into
    mainstream performance metrics: latency (avg / p95 ms), token volume,
    LLM call count, and cost."""
    perfs = [r.get("perf") or {} for r in results if r.get("perf")]
    n = len(perfs)

    def _sum(key: str) -> float:
        return sum(float(p.get(key) or 0) for p in perfs)

    def _avg(key: str) -> float:
        vals = [float(p[key]) for p in perfs if p.get(key)]
        return round(sum(vals) / len(vals), 2) if vals else 0.0

    def _p95(key: str) -> float:
        vals = sorted(float(p[key]) for p in perfs if p.get(key))
        if not vals:
            return 0.0
        return round(vals[min(len(vals) - 1, int(len(vals) * 0.95) - 1)], 2)

    tokens = _sum("total_tokens")
    cost = _sum("cost_usd")
    return {
        "tasks_with_perf": n,
        "llm_calls": _sum("llm_calls"),
        "prompt_tokens": _sum("prompt_tokens"),
        "completion_tokens": _sum("completion_tokens"),
        "total_tokens": tokens,
        "cost_usd": round(cost, 4),
        "avg_latency_ms": _avg("avg_latency_ms"),
        "p95_latency_ms": _p95("p95_latency_ms"),
        "avg_tokens_per_task": round(tokens / n, 1) if n else 0.0,
        "cost_per_task": round(cost / n, 4) if n else 0.0,
    }


def compute_stats(results: list[dict]) -> dict:
    passed = [r for r in results if is_pass(r)]
    perf = _perf_stats(results)
    return {
        "total": len(results),
        "passed": len(passed),
        "pass_at_1": round(len(passed) / len(results), 4) if results else 0.0,
        "by_category": group_pass_rate(results, "category"),
        "by_difficulty": group_pass_rate(results, "difficulty"),
        "failure_distribution": failure_distribution(results),
        "avg_duration_s": average(results, "duration_s"),
        # Prefer the LLM-trace perf (real tokens per task) over the legacy
        # `token_usage` field (the worker never populated it).
        "avg_tokens": perf["avg_tokens_per_task"]
        if perf["tasks_with_perf"]
        else average(results, "token_usage"),
        "performance": perf,
        "cost_per_solved": round(perf["cost_usd"] / len(passed), 4) if passed else 0.0,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def compute_comparison(base_results: list[dict], treat_results: list[dict]) -> dict:
    """Baseline vs treatment delta report — the "提升多少" quantitative answer.

    Used by `scorer --compare BASE_DIR TREAT_DIR`: run the same dataset under
    two variants (e.g. `--tag baseline` vs `--tag agentic`) and get the Δ.
    """
    base = compute_stats(base_results)
    treat = compute_stats(treat_results)

    def _group(base_g: dict, treat_g: dict) -> dict:
        keys = sorted(set(base_g) | set(treat_g))
        out = {}
        for k in keys:
            bg = base_g.get(k, {})
            tg = treat_g.get(k, {})
            out[k] = {
                "base_pass_rate": bg.get("pass_rate", 0.0),
                "treatment_pass_rate": tg.get("pass_rate", 0.0),
                "delta": round(tg.get("pass_rate", 0.0) - bg.get("pass_rate", 0.0), 4),
            }
        return out

    def _snap(s: dict) -> dict:
        return {
            "pass_at_1": s["pass_at_1"],
            "passed": s["passed"],
            "total": s["total"],
            "avg_duration_s": s["avg_duration_s"],
            "avg_tokens": s["avg_tokens"],
        }

    return {
        "baseline": _snap(base),
        "treatment": _snap(treat),
        "delta_pass_at_1": round(treat["pass_at_1"] - base["pass_at_1"], 4),
        "delta_avg_duration_s": round(treat["avg_duration_s"] - base["avg_duration_s"], 2),
        "delta_avg_tokens": round(treat["avg_tokens"] - base["avg_tokens"], 2),
        "by_category": _group(base["by_category"], treat["by_category"]),
        "by_difficulty": _group(base["by_difficulty"], treat["by_difficulty"]),
    }


def render_comparison_markdown(cmp: dict, base_label: str, treat_label: str) -> str:
    """Human-readable delta report for the baseline-vs-treatment comparison."""
    lines = [
        "# Eval Comparison",
        "",
        f"- **Baseline:** {base_label} · **Treatment:** {treat_label}",
        f"- **Δ Pass@1:** **{cmp['delta_pass_at_1'] * 100:+.1f}%** "
        f"({cmp['baseline']['passed']}/{cmp['baseline']['total']} → "
        f"{cmp['treatment']['passed']}/{cmp['treatment']['total']})",
        f"- **Δ avg duration:** {cmp['delta_avg_duration_s']:+.1f}s  ·  "
        f"**Δ avg tokens:** {cmp['delta_avg_tokens']:+.0f}",
        "",
        "## Δ Pass@1 by category",
        _table(
            ["category", "base", "treatment", "Δ"],
            [
                [k, f"{v['base_pass_rate'] * 100:.1f}%", f"{v['treatment_pass_rate'] * 100:.1f}%", f"{v['delta'] * 100:+.1f}%"]
                for k, v in cmp["by_category"].items()
            ],
        ),
        "",
        "## Δ Pass@1 by difficulty",
        _table(
            ["difficulty", "base", "treatment", "Δ"],
            [
                [k, f"{v['base_pass_rate'] * 100:.1f}%", f"{v['treatment_pass_rate'] * 100:.1f}%", f"{v['delta'] * 100:+.1f}%"]
                for k, v in cmp["by_difficulty"].items()
            ],
        ),
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- report
def _table(header: list[str], rows: list[list]) -> str:
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(header)]
    line = "| " + " | ".join(str(h).ljust(w) for h, w in zip(header, widths)) + " |"
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    body = ["| " + " | ".join(str(c).ljust(w) for c, w in zip(row, widths)) + " |" for row in rows]
    return "\n".join([line, sep, *body])


def render_markdown(stats: dict, results: list[dict], env: dict) -> str:
    lines: list[str] = []
    lines.append("# MyCoder Eval Report")
    lines.append("")
    lines.append(f"- **Date:** {stats['generated_at']}")
    lines.append(f"- **Environment:** Python {env['python']}, OS {env['os']}, base_url {env['base_url']}")
    lines.append(f"- **Dataset:** {env['dataset']}")
    lines.append(f"- **Total Pass@1:** **{stats['pass_at_1'] * 100:.1f}%** ({stats['passed']}/{stats['total']})")
    lines.append(f"- **Avg duration:** {stats['avg_duration_s']}s  ·  **Avg tokens/task:** {stats['avg_tokens']}")
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

    perf = stats.get("performance") or {}
    if perf.get("tasks_with_perf"):
        lines.append("## Performance (LLM trace)")
        lines.append(_table(
            ["metric", "value"],
            [
                ["LLM calls", str(perf["llm_calls"])],
                ["prompt / completion / total tokens", f"{perf['prompt_tokens']} / {perf['completion_tokens']} / {perf['total_tokens']}"],
                ["avg latency (ms)", str(perf["avg_latency_ms"])],
                ["p95 latency (ms)", str(perf["p95_latency_ms"])],
                ["total cost (USD)", str(perf["cost_usd"])],
                ["cost / task (USD)", str(perf["cost_per_task"])],
                ["avg tokens / task", str(perf["avg_tokens_per_task"])],
                ["cost / solved (USD)", str(stats.get("cost_per_solved", 0.0))],
            ],
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
    table = Table(title=f"MyCoder Eval — Pass@1 {stats['pass_at_1'] * 100:.1f}%")
    for col in ("dimension", "value", "pass", "total", "rate"):
        table.add_column(col)
    for label, group in (("category", stats["by_category"]), ("difficulty", stats["by_difficulty"])):
        for k, v in group.items():
            table.add_row(label, k, str(v["pass"]), str(v["total"]), f"{v['pass_rate'] * 100:.1f}%")
    table.add_row("overall", "", str(stats["passed"]), str(stats["total"]), f"{stats['pass_at_1'] * 100:.1f}%")
    console.print(table)
    print(f"Avg duration: {stats['avg_duration_s']}s  ·  Avg tokens/task: {stats['avg_tokens']}")


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
    plt.title("MyCoder Eval — Pass@1 by category")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval_bench.scorer", description=__doc__)
    parser.add_argument("--results", help="results/<run> directory containing raw_results.json")
    parser.add_argument("--compare", nargs=2, metavar=("BASE_DIR", "TREAT_DIR"),
                        help="baseline vs treatment delta report (two run dirs)")
    parser.add_argument("--chart", action="store_true", help="also render chart.png (needs matplotlib)")
    parser.add_argument("--base-url", default="", help="API base URL recorded in the report")
    parser.add_argument("--badcases", default=None, help="badcase 回流 output path (default <results>/badcases.json)")
    args = parser.parse_args(argv)

    if args.compare:
        base_dir, treat_dir = (Path(d) for d in args.compare)
        base_raw, treat_raw = base_dir / "raw_results.json", treat_dir / "raw_results.json"
        if not base_raw.exists() or not treat_raw.exists():
            print("[scorer] --compare needs raw_results.json in both dirs")
            return 1
        cmp = compute_comparison(load_results(base_raw), load_results(treat_raw))
        (treat_dir / "comparison.json").write_text(
            json.dumps(cmp, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (treat_dir / "comparison.md").write_text(
            render_comparison_markdown(cmp, base_dir.name, treat_dir.name), encoding="utf-8"
        )
        print(f"[compare] Δ Pass@1 = {cmp['delta_pass_at_1'] * 100:+.1f}%  "
              f"(base {cmp['baseline']['pass_at_1'] * 100:.1f}% -> treat {cmp['treatment']['pass_at_1'] * 100:.1f}%)")
        for k, v in cmp["by_category"].items():
            print(f"  {k:12s} {v['base_pass_rate'] * 100:5.1f}% -> {v['treatment_pass_rate'] * 100:5.1f}%  ({v['delta'] * 100:+.1f}%)")
        print(f"[compare] comparison.md/json written to {treat_dir}")
        return 0

    if not args.results:
        print("[scorer] need --results DIR or --compare BASE_DIR TREAT_DIR")
        return 1
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

    badcase_path = Path(args.badcases) if args.badcases else results_dir / "badcases.json"
    n_bad = write_badcases(results, badcase_path)
    print(f"[scorer] badcases -> {badcase_path} ({n_bad} cases)")

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
