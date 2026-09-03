"""OpenRouter performance benchmarks for P1-4 and P1-5.

P1-4 compares the three explicit reasoning strategies using the same model and
task matrix. P1-5 measures conflict detection and the downstream answer change
before/after a gold-labelled polluted memory is deprecated.

This is an online benchmark: it requires ``OPENROUTER_API_KEY`` and sends real
requests to OpenRouter. Unit tests inject a fake LLM and never use the network.

Usage:
    OPENROUTER_API_KEY=... python -m eval_bench.p1_openrouter_perf
    python -m eval_bench.p1_openrouter_perf --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from mycoder.llm import LLM
from mycoder.memory import MemoryEntry, MemoryStore
from mycoder.memory.maintenance import MemoryMaintainer
from mycoder.prompts.reasoning import REASONING_STRATEGIES, build_reasoning_section

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "minimax/minimax-m3:free"

REASONING_CASES = [
    {
        "id": "architecture",
        "expected_strategy": "plan_execute",
        "task": (
            "为一个单体 Python Agent 设计向多 Agent 架构的渐进迁移方案。"
            "答案必须覆盖模块边界、依赖顺序、灰度/回滚和测试验证。"
        ),
        "requirements": [
            ["模块", "边界"],
            ["依赖", "顺序"],
            ["回滚", "灰度"],
            ["测试", "验证"],
        ],
    },
    {
        "id": "debugging",
        "expected_strategy": "reflection",
        "task": (
            "一个流式 Agent 偶发重复执行写工具。请给出排查和修复方案，"
            "必须覆盖复现、幂等、部分流失败和回归测试。"
        ),
        "requirements": [
            ["复现"],
            ["幂等"],
            ["部分", "流"],
            ["回归", "测试"],
        ],
    },
    {
        "id": "tool_choice",
        "expected_strategy": "react",
        "task": (
            "你需要确认 config/model_routing.yaml 中 OpenRouter 的 fast 模型。"
            "说明最短的观察与行动路径，覆盖读取配置、定位 provider 和报告结果。"
        ),
        "requirements": [
            ["读取", "配置"],
            ["provider", "OpenRouter"],
            ["fast"],
            ["结果", "报告"],
        ],
    },
]

MEMORY_CASES = [
    {
        "id": "jwt_ttl",
        "correct": "CoreCoder auth-service 的 JWT access token 有效期配置为 15 分钟。",
        "polluted": "CoreCoder auth-service 的 JWT access token 有效期配置为 120 分钟。",
        "unrelated": "CoreCoder CI 流水线使用 pytest 执行单元测试。",
        "question": "CoreCoder 的 JWT access token 有效期是多少？",
        "expected": ["15 分钟", "15分钟"],
        "forbidden": ["120 分钟", "120分钟", "两小时", "2 小时"],
    },
    {
        "id": "database_port",
        "correct": "CoreCoder metadata-service 的 PostgreSQL 端口配置为 5432。",
        "polluted": "CoreCoder metadata-service 的 PostgreSQL 端口配置为 3306。",
        "unrelated": "CoreCoder metadata-service 使用 Alembic 管理数据库迁移。",
        "question": "metadata-service 的 PostgreSQL 端口是多少？",
        "expected": ["5432"],
        "forbidden": ["3306"],
    },
    {
        "id": "python_version",
        "correct": "CoreCoder sandbox-runtime 的 Python 版本固定为 3.12。",
        "polluted": "CoreCoder sandbox-runtime 的 Python 版本固定为 3.9。",
        "unrelated": "CoreCoder sandbox-runtime 默认禁止容器网络访问。",
        "question": "sandbox-runtime 固定使用哪个 Python 版本？",
        "expected": ["3.12"],
        "forbidden": ["3.9"],
    },
]

_MARKERS = {
    "react": "thought:",
    "plan_execute": "plan:",
    "reflection": "reflection:",
}


def build_openrouter_llm(
    *, api_key: str | None = None, model: str = DEFAULT_MODEL, max_tokens: int = 700
) -> LLM:
    """Build the real OpenRouter client; keys are accepted from env, not CLI."""
    key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is required for the online benchmark")
    return LLM(
        model=model,
        api_key=key,
        base_url=OPENROUTER_BASE_URL,
        provider="openrouter",
        temperature=0,
        max_tokens=max_tokens,
        caller="p1_openrouter_perf",
    )


def measure_chat(llm, messages: list[dict]) -> dict:
    """One streamed call with wall latency, TTFT, token usage and safe error."""
    started = time.monotonic()
    first_token_at: float | None = None

    def on_token(_token: str) -> None:
        nonlocal first_token_at
        if first_token_at is None:
            first_token_at = time.monotonic()

    try:
        response = llm.chat(messages, on_token=on_token)
    except Exception as exc:  # noqa: BLE001 - one bad sample must not abort a run
        status_code = getattr(exc, "status_code", None)
        return {
            "status": "error",
            "error_type": type(exc).__name__,
            "status_code": status_code,
            "error": _safe_error(exc),
            "latency_ms": round((time.monotonic() - started) * 1000, 2),
            "ttft_ms": None,
            "content": "",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
        }
    finished = time.monotonic()
    return {
        "status": "success",
        "latency_ms": round((finished - started) * 1000, 2),
        "ttft_ms": (
            round((first_token_at - started) * 1000, 2)
            if first_token_at is not None
            else None
        ),
        "content": response.content,
        "prompt_tokens": int(response.prompt_tokens or 0),
        "completion_tokens": int(response.completion_tokens or 0),
        "reasoning_tokens": int(response.reasoning_tokens or 0),
    }


def _safe_error(exc: Exception) -> str:
    """Keep a useful failure reason without persisting provider/user IDs."""
    status_code = getattr(exc, "status_code", None)
    body = getattr(exc, "body", None)
    message = None
    if isinstance(body, dict):
        error = body.get("error", body)
        if isinstance(error, dict):
            message = error.get("message")
    if status_code is not None:
        return f"HTTP {status_code}: {message or type(exc).__name__}"[:200]
    text = re.sub(r"\buser_[A-Za-z0-9]+\b", "[redacted-user]", str(exc))
    return text[:200]


def _coverage(content: str, requirements: list[list[str]]) -> float:
    if not requirements:
        return 1.0
    hit = sum(1 for alternatives in requirements if any(term in content for term in alternatives))
    return round(hit / len(requirements), 4)


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)], 2)


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 2) if values else None


def _aggregate_calls(records: list[dict], group_key: str) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[str(record[group_key])].append(record)
    output = {}
    for name, rows in groups.items():
        successful = [row for row in rows if row["status"] == "success"]
        latencies = [float(row["latency_ms"]) for row in successful]
        ttfts = [float(row["ttft_ms"]) for row in successful if row["ttft_ms"] is not None]
        output[name] = {
            "calls": len(rows),
            "success_rate": round(len(successful) / len(rows), 4),
            "avg_latency_ms": _mean(latencies),
            "p95_latency_ms": _p95(latencies),
            "avg_ttft_ms": _mean(ttfts),
            "p95_ttft_ms": _p95(ttfts),
            "avg_prompt_tokens": _mean([row["prompt_tokens"] for row in successful]),
            "avg_completion_tokens": _mean(
                [row["completion_tokens"] for row in successful]
            ),
            "avg_reasoning_tokens": _mean(
                [row["reasoning_tokens"] for row in successful]
            ),
            "requirement_coverage": _mean(
                [row.get("requirement_coverage", 0.0) for row in successful]
            ),
            "marker_adherence_rate": (
                round(
                    sum(bool(row.get("marker_adherent")) for row in successful)
                    / len(successful),
                    4,
                )
                if successful
                else 0.0
            ),
        }
    return output


def run_reasoning_benchmark(
    llm,
    *,
    cases: list[dict] | None = None,
    repeats: int = 1,
    request_delay: float = 0.0,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """P1-4: fixed task matrix x every reasoning strategy."""
    cases = cases or REASONING_CASES
    records = []
    for repeat in range(repeats):
        for case in cases:
            for strategy in REASONING_STRATEGIES:
                system = (
                    "你是严谨的 coding agent 性能测试对象。答案不超过 350 字。"
                    + build_reasoning_section(strategy)
                )
                measured = measure_chat(
                    llm,
                    [
                        {"role": "system", "content": system},
                        {"role": "user", "content": case["task"]},
                    ],
                )
                content = measured.pop("content")
                measured.update(
                    {
                        "case_id": case["id"],
                        "strategy": strategy,
                        "expected_strategy": case["expected_strategy"],
                        "is_expected_strategy": strategy == case["expected_strategy"],
                        "repeat": repeat + 1,
                        "marker_adherent": _MARKERS[strategy] in content.lower(),
                        "requirement_coverage": _coverage(
                            content, case["requirements"]
                        ),
                        "output": content[:1000],
                    }
                )
                records.append(measured)
                if request_delay:
                    sleep(request_delay)
    successful = [record for record in records if record["status"] == "success"]
    expected = [record for record in successful if record["is_expected_strategy"]]
    return {
        "suite": "p1_4_reasoning_strategies",
        "case_count": len(cases),
        "repeats": repeats,
        "call_count": len(records),
        "successful_calls": len(successful),
        "by_strategy": _aggregate_calls(records, "strategy"),
        "expected_strategy": {
            "calls": len(expected),
            "avg_requirement_coverage": _mean(
                [record["requirement_coverage"] for record in expected]
            ),
            "avg_latency_ms": _mean([record["latency_ms"] for record in expected]),
            "avg_completion_tokens": _mean(
                [record["completion_tokens"] for record in expected]
            ),
        },
        "records": records,
    }


def _answer_is_correct(content: str, case: dict) -> bool:
    return any(term in content for term in case["expected"]) and not any(
        term in content for term in case["forbidden"]
    )


def _memory_messages(memories: list[MemoryEntry], question: str) -> list[dict]:
    context = "\n".join(f"- {entry.content}" for entry in memories)
    return [
        {
            "role": "system",
            "content": (
                "你是记忆问答评测对象。只能依据给定记忆回答；若记忆冲突，"
                "明确指出冲突。答案只写结论，不补充外部知识。"
            ),
        },
        {"role": "user", "content": f"记忆：\n{context}\n\n问题：{question}"},
    ]


def _conflict_pairs(issues: list[dict]) -> set[tuple[str, str]]:
    pairs = set()
    for issue in issues:
        related = issue.get("related_id")
        if issue.get("issue") == "conflicting" and related:
            pairs.add(tuple(sorted((str(issue["id"]), str(related)))))
    return pairs


def run_memory_benchmark(
    llm,
    *,
    cases: list[dict] | None = None,
    base_dir: str | Path | None = None,
    request_delay: float = 0.0,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """P1-5: conflict detection plus OpenRouter answer quality before/after."""
    cases = cases or MEMORY_CASES
    temp = tempfile.TemporaryDirectory(prefix="mycoder-p1-memory-") if base_dir is None else None
    root = Path(base_dir or temp.name)  # type: ignore[union-attr]
    records = []
    try:
        for case in cases:
            case_root = root / case["id"]
            store = MemoryStore(
                project_dir=case_root / "project",
                global_dir=case_root / "global",
                embedder=None,
            )
            try:
                correct_id = store.save(
                    MemoryEntry(content=case["correct"], confidence=0.9), dedup=False
                )
                polluted_id = store.save(
                    MemoryEntry(content=case["polluted"], confidence=0.6), dedup=False
                )
                store.save(
                    MemoryEntry(content=case["unrelated"], confidence=0.8), dedup=False
                )
                before = measure_chat(
                    llm,
                    _memory_messages(store.list(limit=1000), case["question"]),
                )
                if request_delay:
                    sleep(request_delay)

                maintainer = MemoryMaintainer(store)
                audit_started = time.monotonic()
                issues = maintainer.audit_integrity()
                audit_ms = round((time.monotonic() - audit_started) * 1000, 3)
                found_pairs = _conflict_pairs(issues)
                expected_pairs = {tuple(sorted((correct_id, polluted_id)))}

                correction_started = time.monotonic()
                corrected = maintainer.correct_memory(
                    polluted_id, reason="benchmark_gold_label_pollution"
                )
                correction_ms = round(
                    (time.monotonic() - correction_started) * 1000, 3
                )
                after = measure_chat(
                    llm,
                    _memory_messages(store.list(limit=1000), case["question"]),
                )
                if request_delay:
                    sleep(request_delay)

                before_content = before.pop("content")
                after_content = after.pop("content")
                records.append(
                    {
                        "case_id": case["id"],
                        "expected_conflict_pairs": len(expected_pairs),
                        "found_conflict_pairs": len(found_pairs),
                        "true_positive_pairs": len(expected_pairs & found_pairs),
                        "audit_latency_ms": audit_ms,
                        "correction_latency_ms": correction_ms,
                        "correction_applied": corrected,
                        "before": {
                            **before,
                            "correct": _answer_is_correct(before_content, case),
                            "output": before_content[:500],
                        },
                        "after": {
                            **after,
                            "correct": _answer_is_correct(after_content, case),
                            "output": after_content[:500],
                        },
                    }
                )
            finally:
                store.close()
    finally:
        if temp is not None:
            temp.cleanup()

    expected_pairs = sum(row["expected_conflict_pairs"] for row in records)
    found_pairs = sum(row["found_conflict_pairs"] for row in records)
    true_pairs = sum(row["true_positive_pairs"] for row in records)
    before_success = [row for row in records if row["before"]["status"] == "success"]
    after_success = [row for row in records if row["after"]["status"] == "success"]
    before_accuracy = (
        sum(row["before"]["correct"] for row in before_success) / len(before_success)
        if before_success
        else 0.0
    )
    after_accuracy = (
        sum(row["after"]["correct"] for row in after_success) / len(after_success)
        if after_success
        else 0.0
    )
    return {
        "suite": "p1_5_memory_pollution_correction",
        "case_count": len(cases),
        "call_count": len(cases) * 2,
        "correction_mode": "gold-labelled polluted memory is deprecated",
        "conflict_pair_precision": round(true_pairs / found_pairs, 4) if found_pairs else 0.0,
        "conflict_pair_recall": round(true_pairs / expected_pairs, 4) if expected_pairs else 0.0,
        "avg_audit_latency_ms": _mean([row["audit_latency_ms"] for row in records]),
        "avg_correction_latency_ms": _mean(
            [row["correction_latency_ms"] for row in records]
        ),
        "before": {
            "answer_accuracy": round(before_accuracy, 4),
            "avg_latency_ms": _mean(
                [row["before"]["latency_ms"] for row in before_success]
            ),
            "avg_completion_tokens": _mean(
                [row["before"]["completion_tokens"] for row in before_success]
            ),
        },
        "after": {
            "answer_accuracy": round(after_accuracy, 4),
            "avg_latency_ms": _mean(
                [row["after"]["latency_ms"] for row in after_success]
            ),
            "avg_completion_tokens": _mean(
                [row["after"]["completion_tokens"] for row in after_success]
            ),
        },
        "accuracy_delta": round(after_accuracy - before_accuracy, 4),
        "records": records,
    }


def build_report(
    llm,
    *,
    model: str,
    suite: str = "all",
    repeats: int = 1,
    request_delay: float = 0.0,
) -> dict:
    report = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider": "openrouter",
        "base_url": OPENROUTER_BASE_URL,
        "model": model,
        "online": True,
        "methodology": {
            "temperature": 0,
            "streaming": True,
            "reasoning_tokens_are_subset_of_completion": True,
        },
    }
    if suite in {"all", "reasoning"}:
        report["p1_4_reasoning"] = run_reasoning_benchmark(
            llm, repeats=repeats, request_delay=request_delay
        )
    if suite in {"all", "memory"}:
        report["p1_5_memory"] = run_memory_benchmark(
            llm, request_delay=request_delay
        )
    return report


def render_markdown(report: dict) -> str:
    def pct(value) -> str:
        return f"{float(value):.1%}" if value is not None else "n/a"

    def metric(value, suffix: str = "") -> str:
        return f"{value}{suffix}" if value is not None else "n/a"

    lines = [
        "# P1-4 / P1-5 OpenRouter Performance Report",
        "",
        f"- Generated: {report['generated_at']}",
        f"- Provider/model: openrouter / `{report['model']}`",
        "- Reasoning tokens are reported separately but remain part of completion tokens.",
    ]
    reasoning = report.get("p1_4_reasoning")
    if reasoning:
        lines.extend(
            [
                "",
                "## P1-4 Reasoning strategies",
                "",
                f"- Cases/repeats/calls: {reasoning['case_count']} / {reasoning['repeats']} / {reasoning['call_count']}",
                f"- Successful calls: {reasoning['successful_calls']}/{reasoning['call_count']}",
                "- This small run is a smoke baseline, not a production SLO.",
                "",
                "| Strategy | Success | Avg latency | P95 | TTFT | Completion tokens | Coverage | Marker |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name, row in reasoning["by_strategy"].items():
            lines.append(
                f"| {name} | {pct(row['success_rate'])} | {metric(row['avg_latency_ms'], ' ms')} | "
                f"{metric(row['p95_latency_ms'], ' ms')} | {metric(row['avg_ttft_ms'], ' ms')} | "
                f"{metric(row['avg_completion_tokens'])} | {pct(row['requirement_coverage'])} | "
                f"{pct(row['marker_adherence_rate'])} |"
            )
    memory = report.get("p1_5_memory")
    if memory:
        lines.extend(
            [
                "",
                "## P1-5 Memory pollution correction",
                "",
                f"- Gold-labelled cases/calls: {memory['case_count']} / {memory['call_count']}",
                "- Detection identifies conflicting pairs; the benchmark gold label selects which entry to deprecate.",
                f"- Conflict pair precision / recall: {pct(memory['conflict_pair_precision'])} / {pct(memory['conflict_pair_recall'])}",
                f"- Answer accuracy before / after: {pct(memory['before']['answer_accuracy'])} / {pct(memory['after']['answer_accuracy'])}",
                f"- Accuracy delta: {memory['accuracy_delta']:+.1%}",
                f"- Local audit / correction latency: {memory['avg_audit_latency_ms']} ms / {memory['avg_correction_latency_ms']} ms",
            ]
        )
    return "\n".join(lines) + "\n"


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(override=False)
    except ImportError:  # pragma: no cover - project dependency in normal installs
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m eval_bench.p1_openrouter_perf", description=__doc__
    )
    parser.add_argument("--suite", choices=["all", "reasoning", "memory"], default="all")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--request-delay", type=float, default=1.0)
    parser.add_argument("--report", default=None, help="JSON output path")
    parser.add_argument("--dry-run", action="store_true", help="validate cases without API calls")
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.max_tokens < 1 or args.request_delay < 0:
        parser.error("repeats/max-tokens must be positive and request-delay non-negative")

    reasoning_calls = len(REASONING_CASES) * len(REASONING_STRATEGIES) * args.repeats
    memory_calls = len(MEMORY_CASES) * 2
    planned_calls = (
        (reasoning_calls if args.suite in {"all", "reasoning"} else 0)
        + (memory_calls if args.suite in {"all", "memory"} else 0)
    )
    if args.dry_run:
        print(
            f"[dry-run] suite={args.suite} model={args.model} planned_openrouter_calls={planned_calls}"
        )
        return 0

    _load_dotenv()
    try:
        llm = build_openrouter_llm(model=args.model, max_tokens=args.max_tokens)
    except RuntimeError as exc:
        print(f"[p1-perf] {exc}")
        return 2

    report = build_report(
        llm,
        model=args.model,
        suite=args.suite,
        repeats=args.repeats,
        request_delay=args.request_delay,
    )
    output = Path(args.report) if args.report else Path("results/perf") / (
        f"p1-openrouter-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = output.with_suffix(".md")
    markdown.write_text(render_markdown(report), encoding="utf-8")
    successful = 0
    total = 0
    if reasoning := report.get("p1_4_reasoning"):
        successful += reasoning["successful_calls"]
        total += reasoning["call_count"]
    if memory := report.get("p1_5_memory"):
        for record in memory["records"]:
            for phase in ("before", "after"):
                total += 1
                successful += record[phase]["status"] == "success"
    print(f"[p1-perf] OpenRouter calls={successful}/{total} report={output}")
    print(f"[p1-perf] markdown={markdown}")
    return 0 if successful == total else 1


if __name__ == "__main__":
    sys.exit(main())
