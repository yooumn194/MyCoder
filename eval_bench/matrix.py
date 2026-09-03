"""Reproducible 30-task x 3-repeat agent ablation matrix.

Runs the same frozen dataset for three single-agent reasoning strategies and
the provider-routed multi-agent AUTO strategy, then reports mean, population
standard deviation, and failure distributions. No API key is accepted on the
command line; use ``MYCODER_BENCH_API_KEY`` for an authenticated API.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from . import runner
from mycoder.config import Config

VARIANTS: dict[str, dict[str, str]] = {
    "single_react": {
        "execution_mode": "single", "reasoning_strategy": "react",
        "orchestration_strategy": "sequential",
    },
    "single_plan_execute": {
        "execution_mode": "single", "reasoning_strategy": "plan_execute",
        "orchestration_strategy": "sequential",
    },
    "single_reflection": {
        "execution_mode": "single", "reasoning_strategy": "reflection",
        "orchestration_strategy": "sequential",
    },
    "multi_auto": {
        "execution_mode": "multi", "reasoning_strategy": "auto",
        "orchestration_strategy": "auto",
    },
}


def build_plan(
    results_root: Path,
    repeats: int = 3,
    variants: dict[str, dict[str, str]] | None = None,
) -> list[dict]:
    matrix = variants or VARIANTS
    return [
        {
            "variant": name,
            "repeat": repeat,
            "options": options,
            "results": results_root / name / f"repeat-{repeat}",
        }
        for name, options in matrix.items()
        for repeat in range(1, repeats + 1)
    ]


def _metric(records: list[dict], field: str) -> dict:
    values = [float(item[field]) for item in records if item.get(field) is not None]
    return {
        "mean": round(statistics.fmean(values), 4) if values else None,
        "stddev": round(statistics.pstdev(values), 4) if values else None,
        "n": len(values),
    }


def aggregate(plan: list[dict]) -> dict:
    grouped: dict[str, list[list[dict]]] = {}
    for run in plan:
        path = Path(run["results"]) / "raw_results.json"
        if path.exists():
            grouped.setdefault(run["variant"], []).append(
                json.loads(path.read_text(encoding="utf-8"))
            )
    summary: dict[str, dict] = {}
    for variant, repetitions in grouped.items():
        records = [record for repetition in repetitions for record in repetition]
        passed = [
            item for item in records
            if item.get("agent_status") == "success" and item.get("error_class") is None
        ]
        failures = [item for item in records if item not in passed]
        pass_values = [
            sum(
                1 for item in repetition
                if item.get("agent_status") == "success" and item.get("error_class") is None
            ) / len(repetition)
            for repetition in repetitions
            if repetition
        ]
        summary[variant] = {
            "samples": len(records),
            "completed_repeats": len(repetitions),
            "pass_rate": {
                "mean": round(statistics.fmean(pass_values), 4) if pass_values else None,
                "stddev": round(statistics.pstdev(pass_values), 4) if pass_values else None,
                "n": len(pass_values),
            },
            "duration_s": _metric(records, "duration_s"),
            "token_usage": _metric(records, "token_usage"),
            "failure_distribution": dict(Counter(
                item.get("error_class") or item.get("agent_status") or "UNKNOWN"
                for item in failures
            )),
        }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval_bench.matrix", description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--dataset", default=str(Path(__file__).parent / "dataset.json"))
    parser.add_argument("--workspace", default=str(Path(__file__).parent / "workspace"))
    parser.add_argument("--workspace-id", default="default")
    parser.add_argument("--results", default=None)
    parser.add_argument("--parallel", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    dataset_path = Path(args.dataset).resolve()
    data = runner.load_dataset(dataset_path)
    violations = runner.validate_dataset(data)
    if violations:
        print("\n".join(f"[schema] {item}" for item in violations))
        return 1
    if len(data) != 30 and not args.allow_partial:
        print(f"[matrix] expected exactly 30 tasks, got {len(data)}; use --allow-partial")
        return 1
    if args.repeats != 3 and not args.allow_partial:
        print(f"[matrix] expected exactly 3 repeats, got {args.repeats}; use --allow-partial")
        return 1

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    root = Path(args.results or f"results/ablation-{timestamp}").resolve()
    plan = build_plan(root, repeats=args.repeats)
    model_config = Config.from_env()
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "task_count": len(data),
        "repeats": args.repeats,
        "planned_api_runs": len(data) * len(plan),
        "base_url": args.base_url,
        "workspace_id": args.workspace_id,
        "provider": model_config.provider,
        "model": model_config.model,
        "temperature": model_config.temperature,
        "variants": VARIANTS,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.dry_run:
        print(
            f"[dry-run] {len(data)} tasks x {args.repeats} repeats x "
            f"{len(VARIANTS)} variants = {manifest['planned_api_runs']} API runs"
        )
        print(f"[dry-run] manifest -> {root / 'manifest.json'}")
        return 0

    for item in plan:
        options = item["options"]
        code = runner.main([
            "--base-url", args.base_url,
            "--dataset", str(dataset_path),
            "--workspace", args.workspace,
            "--workspace-id", args.workspace_id,
            "--results", str(item["results"]),
            "--parallel", str(args.parallel),
            "--tag", item["variant"],
            "--execution-mode", options["execution_mode"],
            "--reasoning-strategy", options["reasoning_strategy"],
            "--orchestration-strategy", options["orchestration_strategy"],
        ])
        if code != 0:
            return code

    report = {"manifest": manifest, "variants": aggregate(plan)}
    (root / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[done] ablation summary -> {root / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
