"""LLM-as-Judge evaluation — quality scoring beyond pass/fail (P2).

Pass/fail (eval_bench/runner.py) only answers "did the tests pass?"; it gives
no signal on a partially-correct or borderline answer. judge_run.py runs the
LLMJudge over a QA set and produces a quality report:

  * score distribution over the {0.0, 0.5, 1.0} grid (correct / partial / wrong);
  * per-case score + one-line reasoning (for badcase triage);
  * a low-score 回流 list to feed back into the dataset or a future SFT pass.

Best-effort by design: without a judge LLM (no API key) every case scores a
neutral 0.5 with a warning — the eval never crashes because judging failed.

Usage:
    python -m eval_bench.judge_run                        # built-in sample QA
    python -m eval_bench.judge_run --input qa.json --report out/judge_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mycoder.memory.citation import verify_citations
from eval_bench.judge import LLMJudge

# 开箱即用样本：覆盖 1.0（正确完整）/ 0.5（部分）/ 0.0（错误离题）三类，用于
# 快速验证 judge 链路与报告格式。
SAMPLE_QA: list[dict] = [
    {
        "question": "RAG 为什么用混合检索（向量 + BM25）？",
        "reference": "向量检索抓语义相近，BM25 抓精确词命中；两者互补后经 RRF 融合，兼顾召回率与精确匹配。",
        "answer": "混合检索结合向量（语义）与 BM25（关键词精确匹配），用 RRF 融合两者排序，兼顾语义召回和精确命中。",
    },
    {
        "question": "Agent 调用工具超时后敢不敢直接重试？",
        "reference": "取决于幂等性：读操作或幂等写可直接重试；非幂等写（转账、发消息）重试可能重复执行，需幂等键去重或改为人工确认。",
        "answer": "不能直接重试，重试有风险。",
    },
    {
        "question": "上下文窗口满了怎么办？",
        "reference": "分层压缩：先截断超长工具输出，再用 LLM 总结旧对话，紧急时硬折叠只留摘要 + 最近几轮；配合外部记忆避免关键信息丢失。",
        "answer": "可以用更大的模型，上下文窗口更大。",
    },
]


def evaluate(cases: list[dict], judge=None, threshold: float = 0.5) -> dict:
    """Score each case with the LLMJudge and aggregate the quality report.

    ``judge`` is injectable for tests; a ``judge`` with no LLM yields neutral
    0.5s and ``judge_available=False`` so callers can tell real scores apart.
    """
    judge = judge if judge is not None else LLMJudge()
    per_case: list[dict] = []
    for c in cases:
        answer = str(c.get("answer", ""))
        verdict = judge.judge(
            str(c.get("question", "")),
            answer,
            str(c.get("reference") or ""),
        )
        entry = {
            "id": c.get("id", f"case-{len(per_case)}"),
            "question": str(c.get("question", ""))[:200],
            "score": verdict["score"],
            "reasoning": verdict["reasoning"],
        }
        # P2 citation integrity: when the case carries retrieval contexts, the
        # deterministic citation verdict is a scoring dimension alongside the
        # judge's quality score.
        contexts = c.get("contexts")
        if isinstance(contexts, list) and contexts:
            entry["citation"] = verify_citations(answer, len(contexts))
        per_case.append(entry)

    total = len(per_case)
    correct = sum(1 for r in per_case if r["score"] == 1.0)
    partial = sum(1 for r in per_case if r["score"] == 0.5)
    wrong = sum(1 for r in per_case if r["score"] == 0.0)
    low_score = [r for r in per_case if r["score"] < threshold]
    return {
        "total": total,
        "judge_available": judge._llm is not None,
        "correct": correct,
        "partial": partial,
        "wrong": wrong,
        "avg_score": round(sum(r["score"] for r in per_case) / total, 3) if total else 0.0,
        "per_case": per_case,
        # badcase 回流清单：低分 case 带理由，可直接喂回数据集/SFT。
        "low_score_cases": low_score,
    }


def load_qa(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def _append_queue(queue_path: Path, low_score: list[dict]) -> int:
    """Append low-score badcases to a JSONL queue, deduped by id (P2 回流)."""
    existing: set[str] = set()
    if queue_path.exists():
        for line in queue_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                existing.add(json.loads(line).get("id", ""))
            except json.JSONDecodeError:  # pragma: no cover - malformed line
                continue
    added = 0
    with queue_path.open("a", encoding="utf-8") as fh:
        for r in low_score:
            if r["id"] in existing:
                continue
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            existing.add(r["id"])
            added += 1
    return added


def _regenerate(queue_path: Path, out_path: Path) -> int:
    """Read the badcase queue and emit a regeneration manifest for human review.

    The manifest is the bridge into the eval dataset: review each sample, keep
    the ones that should be covered, and fold them into eval_bench/dataset.json
    (or the QA set) so the next full regression covers them (发现→修复→预防).
    """
    samples: list[dict] = []
    if queue_path.exists():
        for line in queue_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:  # pragma: no cover - malformed line
                continue
    manifest = {
        "source": str(queue_path),
        "sample_count": len(samples),
        "review_instruction": (
            "人工审核以下低分样本：保留需纳入回归覆盖的条目，"
            "将 {question, reference, answer} 补全后加入评测数据集。"
        ),
        "samples": samples,
    }
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(samples)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m eval_bench.judge_run", description=__doc__
    )
    parser.add_argument(
        "--input", default=None,
        help="QA JSON file: list of {id?, question, reference?, answer, contexts?}; default built-in sample",
    )
    parser.add_argument("--report", default=None, help="write JSON report to path")
    parser.add_argument("--threshold", type=float, default=0.5, help="low-score 判定阈值 (default 0.5)")
    parser.add_argument(
        "--queue", default=None,
        help="append low-score badcases to this JSONL queue (badcase 回流)",
    )
    parser.add_argument(
        "--regenerate", action="store_true",
        help="read the badcase queue and emit a regeneration manifest for review",
    )
    args = parser.parse_args(argv)

    # P2 badcase 回流：单独模式，读 queue 生成回流 manifest。
    if args.regenerate:
        queue = Path(args.queue) if args.queue else None
        if queue is None or not queue.exists():
            print("[regenerate] need --queue pointing at an existing badcase queue")
            return 1
        manifest_path = queue.with_name("regenerate_manifest.json")
        n = _regenerate(queue, manifest_path)
        print(f"[regenerate] manifest -> {manifest_path} ({n} samples to review)")
        return 0

    cases = load_qa(Path(args.input)) if args.input else SAMPLE_QA
    if not cases:
        print(f"[judge] empty QA set: {args.input or '(built-in)'}")
        return 1

    report = evaluate(cases, threshold=args.threshold)
    if args.report:
        Path(args.report).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    if not report["judge_available"]:
        print("[judge] ⚠ no judge LLM (no API key) — all scores are neutral 0.5")
    print(
        f"[judge] {report['total']} cases  correct={report['correct']} "
        f"partial={report['partial']} wrong={report['wrong']}  "
        f"avg={report['avg_score']:.2f}"
    )
    for r in report["per_case"]:
        mark = {1.0: "✅", 0.5: "◐", 0.0: "❌"}.get(r["score"], "·")
        print(f"  {mark} [{r['score']:.1f}] {r['id']}: {r['reasoning']}")
    if report["low_score_cases"]:
        print(
            f"[judge] {len(report['low_score_cases'])} low-score case(s) -> "
            f"回流清单见报告 low_score_cases"
        )
    if args.queue:
        queue_path = Path(args.queue)
        added = _append_queue(queue_path, report["low_score_cases"])
        print(f"[judge] badcase queue -> {queue_path} (+{added} samples, 下次 --regenerate 回流)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
