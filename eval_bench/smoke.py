"""One-task SWE-bench control smoke test.

This is deliberately a thin preset over the canonical adapter: it fixes one
stable instance and conservative token limits while retaining the adapter's
workspace isolation, patch extraction, and manifest checks.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .runtime_config import snapshot
from .swe_verified import adapter

DEFAULT_INSTANCE = "django__django-11133"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m eval_bench.smoke",
        description=__doc__,
    )
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--instance-id", default=DEFAULT_INSTANCE)
    parser.add_argument("--workspace", type=Path, default=Path("workspaces/swe-bench/local"))
    parser.add_argument("--repo-cache", type=Path, default=Path("workspaces/swe-bench-repos"))
    parser.add_argument("--results", type=Path, default=Path("results/swe-bench/smoke"))
    parser.add_argument("--max-tokens", type=int, default=35_000)
    parser.add_argument("--soft-budget-tokens", type=int, default=20_000)
    parser.add_argument("--timeout-seconds", type=int, default=1_200)
    parser.add_argument("--execution-mode", choices=("single", "multi"), default="single")
    parser.add_argument("--reasoning-strategy", choices=("auto", "react", "plan_execute", "reflection"), default="react")
    parser.add_argument("--orchestration-strategy", choices=("auto", "sequential", "parallel", "conditional"), default="sequential")
    parser.add_argument(
        "--thinking",
        choices=("auto", "enabled", "disabled"),
        default=None,
        help="DeepSeek thinking mode; set the same value when starting the API",
    )
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.soft_budget_tokens >= args.max_tokens:
        parser.error("--soft-budget-tokens must be lower than --max-tokens")
    if args.thinking is not None:
        # Keep the adapter's manifest snapshot aligned with the API process
        # when the caller supplies the same explicit provider setting to both.
        os.environ["MYCODER_DEEPSEEK_THINKING"] = args.thinking
    print(f"[smoke] runtime={snapshot()}")
    forwarded = [
        "--base-url", args.base_url,
        "--instance-id", args.instance_id,
        "--workspace", str(args.workspace),
        "--repo-cache", str(args.repo_cache),
        "--results", str(args.results),
        "--max-tokens", str(args.max_tokens),
        "--soft-budget-tokens", str(args.soft_budget_tokens),
        "--timeout-seconds", str(args.timeout_seconds),
        "--mode", "control",
        "--parallel", "1",
        "--execution-mode", args.execution_mode,
        "--reasoning-strategy", args.reasoning_strategy,
        "--orchestration-strategy", args.orchestration_strategy,
    ]
    for flag, enabled in (("--evaluate", args.evaluate), ("--resume", args.resume), ("--dry-run", args.dry_run)):
        if enabled:
            forwarded.append(flag)
    return adapter.main(forwarded)


if __name__ == "__main__":
    sys.exit(main())
