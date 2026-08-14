"""RAG evaluation (P2) — retrieval metrics on a document + query set.

Answers "RAG 系统如何评测？" with concrete, measurable retrieval metrics:

  * recall@k    — fraction of expected chunk ids present in the top-k results;
  * precision@k— fraction of top-k results that are expected;
  * citation accuracy — fraction of queries whose top-1 chunk is the expected one.

Indexes a document through mycoder/memory/document.py (the hybrid store) and
scores a hand-built query set, where each query declares which chunk indices
should be recalled. Pure stdlib — no server, no LLM, fully runnable offline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mycoder.memory import MemoryStore, HybridRetriever
from mycoder.memory.citation import verify_citations
from mycoder.memory.document import save_document

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

# P2 citation integrity: sample grounded answers to score with verify_citations.
SAMPLE_ANSWERS = [
    {
        "id": "good",
        "answer": "登录使用 JWT[1]，刷新 token 无需重新登录[2]，退款需人工审核[3]。",
        "n_contexts": 3,
    },
    {"id": "hallucinated", "answer": "登录使用 JWT[9]。", "n_contexts": 3},  # [9] 越界 -> 幻觉引用
    {"id": "uncited", "answer": "登录使用 JWT，这里不给出任何来源。", "n_contexts": 3},
]


def _recalled(results: list[dict], terms: list[str], k: int) -> bool:
    """True when every expected term appears in the top-k retrieved content."""
    if not terms:
        return True
    text = " ".join(str(r.get("content", "")) for r in results[:k])
    return all(term in text for term in terms)


def _precision(results: list[dict], terms: list[str], k: int) -> float:
    """Fraction of top-k results that mention any expected term."""
    top = results[:k]
    hit = sum(1 for r in top if any(t in str(r.get("content", "")) for t in terms))
    return hit / len(top) if top else 0.0


def evaluate(
    doc_text: str,
    queries: list[dict],
    *,
    k: int = 3,
    embedder=None,
) -> dict:
    """Index `doc_text` and score `queries` on recall@k / precision / citation.

    recall@k: fraction of queries whose expected terms all appear in top-k;
    precision@k: fraction of top-k chunks that mention any expected term;
    citation accuracy: fraction of queries whose top-1 chunk holds the first
    expected term (i.e. the retriever surfaced the right passage first).

    ``embedder`` defaults to None (BM25-only — deterministic for tests); pass a
    real embedder (e.g. from memory config) to measure semantic-vector recall.
    """
    import tempfile

    store = MemoryStore(
        project_dir=Path(tempfile.mkdtemp()) / "proj",
        global_dir=Path(tempfile.mkdtemp()) / "glob",
        embedder=embedder,
    )
    # Use the production chunk defaults (1500 chars / 15% overlap) — the old
    # hardcoded max_chars=60 shredded a real document into 330 tiny chunks and
    # distorted retrieval. SAMPLE_DOC is short so defaults keep it whole.
    save_document(store, "eval-doc", doc_text, overlap_ratio=0.15)
    retriever = HybridRetriever(store)

    per_query = []
    recalls, precisions, citations = [], [], []
    for q in queries:
        results = retriever.search(q["query"], limit=k)
        terms = list(q.get("expected_terms", []))
        recall = 1.0 if _recalled(results, terms, k) else 0.0
        precision = _precision(results, terms, k)
        top_k = results[:k]
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


class _VectorOnlyRetriever:
    """Pure semantic-vector retrieval (baseline for the attribution table)."""

    def __init__(self, store) -> None:
        self.store = store

    def search(self, query: str, limit: int = 3) -> list[dict]:
        qvec = self.store.embed_query(query)
        if qvec is None:
            return []
        ids_scores = self.store.vector_search(qvec, limit=limit * 3)
        rows = self.store.fetch_rows([mid for mid, _ in ids_scores])
        out = []
        for mid, score in ids_scores:
            row = rows.get(mid)
            if row is not None:
                out.append(
                    {
                        "id": mid,
                        "content": row["content"],
                        "score": float(score),
                        "confidence": float(row.get("confidence", 0.5)),
                        "type": row.get("type"),
                        "scope": row.get("scope"),
                        "source": row.get("source"),
                        "metadata": row.get("metadata"),
                    }
                )
        return out[:limit]


def evaluate_compare(
    doc_text: str,
    queries: list[dict],
    *,
    k: int = 3,
    embedder=None,
) -> dict:
    """Baseline → optimization attribution table on one real eval set.

    Rows, in optimization order:
      * pure_vector  — semantic vectors only (baseline);
      * hybrid_rrf   — + BM25 fused via RRF (混合检索);
      * hybrid_rerank— + RuleReranker on top.

    Each row is (recall@k, precision@k) over the same queries, so the Δ between
    rows is the per-step attribution ("+BM25 混合 → +X%, +Rerank → +Y%").
    Needs a real embedder (pure_vector is meaningless with embedder=None).
    """
    import tempfile

    store = MemoryStore(
        project_dir=Path(tempfile.mkdtemp()) / "proj",
        global_dir=Path(tempfile.mkdtemp()) / "glob",
        embedder=embedder,
    )
    save_document(store, "eval-doc", doc_text, overlap_ratio=0.15)

    def _score(retriever) -> tuple[float, float]:
        recalls, precisions = [], []
        for q in queries:
            results = retriever.search(q["query"], limit=k)
            terms = list(q.get("expected_terms", []))
            recalls.append(1.0 if _recalled(results, terms, k) else 0.0)
            precisions.append(_precision(results, terms, k))
        return (
            round(sum(recalls) / len(recalls), 3) if recalls else 0.0,
            round(sum(precisions) / len(precisions), 3) if precisions else 0.0,
        )

    return {
        "k": k,
        "rows": {
            "pure_vector": _score(_VectorOnlyRetriever(store)),
            "hybrid_rrf": _score(HybridRetriever(store, reranker=None)),
            "hybrid_rerank": _score(HybridRetriever(store)),
        },
    }


def evaluate_citations(answers: list[dict]) -> dict:
    """P2 citation-integrity scoring on sample grounded answers.

    Each sample declares the number of contexts its answer was generated over;
    verify_citations() flags hallucinated references (a cited index outside the
    range) and uncited answers. This is the deterministic half of citation
    scoring — the LLM judge (judge_run) supplies the quality half.
    """
    verdicts = []
    for a in answers:
        v = verify_citations(a.get("answer"), int(a.get("n_contexts", 1)))
        verdicts.append(
            {
                "id": a.get("id", "?"),
                "answer": str(a.get("answer", ""))[:200],
                **v,
            }
        )
    valid = sum(1 for v in verdicts if v["valid"])
    hallucinated = sum(1 for v in verdicts if v["missing"])
    return {
        "citation_integrity": round(valid / len(verdicts), 3) if verdicts else 0.0,
        "hallucinated_reference_ratio": round(hallucinated / len(verdicts), 3)
        if verdicts
        else 0.0,
        "per_answer": verdicts,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval_bench.rag_eval", description=__doc__)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--report", default=None, help="write JSON report to path")
    parser.add_argument(
        "--embedder", choices=["config", "none"], default="none",
        help="'config' uses the memory config embedder (real semantic vectors); "
             "'none' (default) is BM25-only (deterministic)",
    )
    parser.add_argument(
        "--doc", default=None,
        help="document file (markdown/text) to index for a REAL QA eval set",
    )
    parser.add_argument(
        "--queries", default=None,
        help="JSON file: list of {id, query, expected_terms} (or {doc, queries:[...]})",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="run the optimization-attribution table (pure vector → hybrid → +rerank); needs --embedder config",
    )
    args = parser.parse_args(argv)

    embedder = None
    if args.embedder == "config":
        from mycoder.memory.config import load_memory_config
        from mycoder.memory.embedder import get_embedder

        embedder = get_embedder(load_memory_config()["memory"].get("embedder"))

    if args.doc and args.queries:
        doc_text = Path(args.doc).read_text(encoding="utf-8")
        queries = json.loads(Path(args.queries).read_text(encoding="utf-8"))
        if isinstance(queries, dict):
            queries = queries.get("queries", [])
    else:
        doc_text, queries = SAMPLE_DOC, SAMPLE_QUERIES

    if args.compare:
        if embedder is None:
            print("[compare] 需要 --embedder config（纯向量基线需要语义向量）")
            return 1
        table = evaluate_compare(doc_text, queries, k=args.k, embedder=embedder)
        rows = table["rows"]
        names = {
            "pure_vector": "纯向量 (baseline)",
            "hybrid_rrf": "+ BM25 混合(RRF)",
            "hybrid_rerank": "+ Rerank",
        }
        print(f"[compare] 检索优化归因（同一真实评测集, k={args.k}）")
        prev = None
        for key, label in names.items():
            rec, prec = rows[key]
            delta = ""
            if prev is not None:
                delta = f"  (Δ recall {rec - prev[0]:+.1%}, Δ prec {prec - prev[1]:+.1%})"
            print(f"  {label:22s} recall@k={rec:.1%}  precision@k={prec:.1%}{delta}")
            prev = (rec, prec)
        return 0

    report = evaluate(doc_text, queries, k=args.k, embedder=embedder)
    report["citation_integrity"] = evaluate_citations(SAMPLE_ANSWERS)
    if args.report:
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    mode = "semantic-vector" if args.embedder == "config" else "bm25-only"
    print(f"[rag_eval] backend={mode}  k={args.k}")
    print(
        f"recall@{args.k}={report['recall@k']:.1%}  precision@{args.k}={report['precision@k']:.1%}  "
        f"citation_accuracy={report['citation_accuracy']:.1%}  "
        f"citation_integrity={report['citation_integrity']['citation_integrity']:.1%}"
    )
    for q in report["per_query"]:
        print(
            f"  {q['query']:16s} recall@k={q['recall@k']:.0%} "
            f"precision@k={q['precision@k']:.0%} cite={q['citation_accuracy']:.0%}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
