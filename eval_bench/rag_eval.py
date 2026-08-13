"""RAG evaluation (P2) — retrieval metrics on a document + query set.

Answers "RAG 系统如何评测？" with concrete, measurable retrieval metrics:

  * recall@k    — fraction of expected chunk ids present in the top-k results;
  * precision@k— fraction of top-k results that are expected;
  * citation accuracy — fraction of queries whose top-1 chunk is the expected one.

Indexes a document through corecoder/memory/document.py (the hybrid store) and
scores a hand-built query set, where each query declares which chunk indices
should be recalled. Pure stdlib — no server, no LLM, fully runnable offline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from corecoder.memory import MemoryStore, HybridRetriever
from corecoder.memory.document import save_document

SAMPLE_DOC = """\
# 认证模块
## 登录
登录流程使用 JWT token 进行身份验证，token 有效期一小时。
用户每次登录都会获得新的 token，旧 token 立即失效。

## 刷新
刷新 token 可以签发新的访问令牌，不需要重新登录。
刷新接口同样需要校验用户身份。

# 支付模块
## 退款
退款需要人工审核，超过 24 小时未处理自动关闭。
退款金额原路退回，可能收取手续费。

## 结算
结算金额四舍五入到分，每日凌晨统一结算。
"""

SAMPLE_QUERIES = [
    {"query": "JWT 登录", "expected_terms": ["JWT", "登录"]},
    {"query": "刷新 令牌", "expected_terms": ["刷新", "令牌"]},
    {"query": "退款 人工审核", "expected_terms": ["退款", "人工审核"]},
]


def _recalled(results: list[dict], terms: list[str], k: int) -> bool:
    """True when every expected term appears in the top-k retrieved content."""
    if not terms:
        return True
    text = " ".join(str(r.get("content", "")) for r in results[:k])
    return all(term in text for term in terms)


def evaluate(doc_text: str, queries: list[dict], *, k: int = 3) -> dict:
    """Index `doc_text` and score `queries` on recall@k / precision / citation.

    recall@k: fraction of queries whose expected terms all appear in top-k;
    precision@k: fraction of top-k chunks that mention any expected term;
    citation accuracy: fraction of queries whose top-1 chunk holds the first
    expected term (i.e. the retriever surfaced the right passage first).
    """
    import tempfile

    store = MemoryStore(
        project_dir=Path(tempfile.mkdtemp()) / "proj",
        global_dir=Path(tempfile.mkdtemp()) / "glob",
        embedder=None,
    )
    save_document(store, "eval-doc", doc_text, max_chars=60, overlap_ratio=0.0)
    retriever = HybridRetriever(store)

    per_query = []
    recalls, precisions, citations = [], [], []
    for q in queries:
        results = retriever.search(q["query"], limit=k)
        terms = list(q.get("expected_terms", []))
        recall = 1.0 if _recalled(results, terms, k) else 0.0
        top_k = results[:k]
        hit = sum(1 for r in top_k if any(t in str(r.get("content", "")) for t in terms))
        precision = hit / len(top_k) if top_k else 0.0
        top = top_k[0].get("content", "") if top_k else ""
        cite = 1.0 if (terms and terms[0] in top) else 0.0
        recalls.append(recall)
        precisions.append(precision)
        citations.append(cite)
        per_query.append(
            {
                "query": q["query"],
                "recall@k": round(recall, 3),
                "precision@k": round(precision, 3),
                "citation_accuracy": cite,
            }
        )

    return {
        "recall@k": round(sum(recalls) / len(recalls), 3),
        "precision@k": round(sum(precisions) / len(precisions), 3),
        "citation_accuracy": round(sum(citations) / len(citations), 3),
        "per_query": per_query,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval_bench.rag_eval", description=__doc__)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--report", default=None, help="write JSON report to path")
    args = parser.parse_args(argv)

    report = evaluate(SAMPLE_DOC, SAMPLE_QUERIES, k=args.k)
    if args.report:
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"recall@{args.k}={report['recall@k']:.1%}  precision@{args.k}={report['precision@k']:.1%}  "
        f"citation_accuracy={report['citation_accuracy']:.1%}"
    )
    for q in report["per_query"]:
        print(f"  {q['query']:16s} recall={q['recall@k']:.0%} top={q['top_chunk']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
