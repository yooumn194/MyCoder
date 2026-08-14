"""HybridRetriever: BM25 (FTS5) + vector cosine fused with Reciprocal Rank
Fusion (RRF, k=60).

Pipeline:

    query --tokenize--> bm25 top-(limit*3)     (both dbs, score 0..1)
         --embed------> vector top-(limit*3)   (cosine distance)
         --RRF fuse--> ranked ids
         --filters-->  final rows (scope/types/min_conf/deprecated)

When the vector path is unavailable (no embedder / backend degraded), the
result is BM25-only — never an error.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .store import MemoryStore
from .tokenizer import tokenize_chinese

RRF_K = 60


class Reranker(ABC):
    """Re-orders results after RRF fusion. Pluggable so a cross-encoder can
    replace the default rule-based reranker without touching the retriever."""

    @abstractmethod
    def rerank(self, query: str, results: list[dict]) -> list[dict]:
        """Return the same results in a new order (no add/remove)."""


class RuleReranker(Reranker):
    """Zero-dependency reranker: boosts chunks that contain more query terms
    (lexical hit count) and penalizes chunks whose length is far from an ideal
    range — a RAG classic to counter "longest doc always wins" and to nudge
    document chunks over whole-document matches."""

    def __init__(
        self,
        hit_bonus: float = 0.15,
        ideal_min: int = 120,
        ideal_max: int = 1600,
        length_penalty_max: float = 0.3,
    ) -> None:
        self.hit_bonus = hit_bonus
        self.ideal_min = ideal_min
        self.ideal_max = ideal_max
        self.length_penalty_max = length_penalty_max

    def _length_penalty(self, length: int) -> float:
        if length < self.ideal_min:
            return self.length_penalty_max * (self.ideal_min - length) / self.ideal_min
        if length > self.ideal_max:
            return self.length_penalty_max * min(
                2.0, (length - self.ideal_max) / self.ideal_max
            )
        return 0.0

    def rerank(self, query: str, results: list[dict]) -> list[dict]:
        terms = set(tokenize_chinese(query).split())

        def score(result: dict) -> float:
            content = result.get("content") or ""
            hits = sum(1 for term in terms if term in content)
            return (
                float(result.get("score", 0.0))
                + self.hit_bonus * hits
                - self._length_penalty(len(content))
            )

        # stable sort: ties keep the pre-rerank (RRF) order
        return sorted(results, key=score, reverse=True)


class HybridRetriever:
    def __init__(
        self,
        store: MemoryStore,
        *,
        rrf_k: int = RRF_K,
        reranker: Reranker | None = None,
        query_rewriter=None,
    ):
        self.store = store
        self.rrf_k = rrf_k
        self.reranker = reranker if reranker is not None else RuleReranker()
        # P0 (memory/query_rewrite.py): optional multi-turn query rewriter.
        # Only applied when `history` is passed to search; None = no rewrite.
        self.query_rewriter = query_rewriter

    # ------------------------------------------------------------------ public
    def search(
        self,
        query: str,
        limit: int = 10,
        scope: str | None = None,
        types: set[str] | list[str] | None = None,
        min_conf: float = 0.1,
        rerank: bool = True,
        history: list[dict] | None = None,
    ) -> list[dict]:
        """Hybrid search. Returns a list of result dicts (id/content/type/
        scope/source/confidence/score), most relevant first.

        When `rerank` is on, up to 3×limit candidates are fused + filtered and
        then re-ordered by the configured Reranker (default RuleReranker) before
        the limit is applied — so a chunk that the RRF missed on rank but
        matches more query terms can still surface.

        When a `query_rewriter` is configured and `history` is supplied, the
        raw message is first rewritten into a standalone retrieval query (so
        fragments like "那它呢？" resolve against earlier turns).
        """
        if not query or not query.strip():
            return []
        if history is not None and self.query_rewriter is not None:
            query = self.query_rewriter.rewrite(query, history)
            if not query or not query.strip():
                return []
        bm25 = self.store.bm25_search(query, limit=limit * 3, scope=scope)
        vector: list[tuple[str, float]] = []
        qvec = self.store.embed_query(query)
        if qvec is not None:
            vector = self.store.vector_search(qvec, limit=limit * 3, scope=scope)

        fused = self._rrf_fuse(bm25, vector) if vector else bm25
        candidate_limit = limit * 3 if (rerank and self.reranker is not None) else limit
        candidates = self._apply_filters(fused, candidate_limit, scope, types, min_conf)
        if rerank and self.reranker is not None and len(candidates) > 1:
            candidates = self.reranker.rerank(query, candidates)
        return candidates[:limit]

    # ------------------------------------------------------------------ pieces
    def _rrf_fuse(
        self, *ranked_lists: list[tuple[str, float]]
    ) -> list[tuple[str, float]]:
        """Reciprocal-rank fusion: each ranked list contributes 1/(k+rank);
        rank is 1-based positional order within its list."""
        scores: dict[str, float] = {}
        for ranked in ranked_lists:
            for rank, (mem_id, _score) in enumerate(ranked, start=1):
                scores[mem_id] = scores.get(mem_id, 0.0) + 1.0 / (self.rrf_k + rank)
        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    def _apply_filters(
        self,
        fused: list[tuple[str, float]],
        limit: int,
        scope: str | None,
        types: set[str] | list[str] | None,
        min_conf: float,
    ) -> list[dict]:
        if not fused:
            return []
        ids = [mem_id for mem_id, _ in fused]
        # Single batched fetch (IN clause) — avoids an N+1 SELECT per id.
        rows = self.store.fetch_rows(ids)
        type_set = set(types) if types else None
        kept: list[dict] = []
        for mem_id, score in fused:
            row = rows.get(mem_id)
            if row is None:
                continue
            if row.get("deprecated_by"):
                continue
            if scope and row.get("scope") != scope:
                continue
            if type_set and row.get("type") not in type_set:
                continue
            if float(row.get("confidence", 0.0)) < min_conf:
                continue
            kept.append(
                {
                    "id": mem_id,
                    "content": row["content"],
                    "type": row["type"],
                    "scope": row["scope"],
                    "source": row["source"],
                    "confidence": float(row["confidence"]),
                    "score": float(score),
                    "metadata": row.get("metadata"),
                }
            )
            if len(kept) >= limit:
                break
        # searching a memory counts as an access (feeds maintenance decay)
        self.store.mark_accessed([r["id"] for r in kept])
        return kept
