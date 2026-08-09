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

from .store import MemoryStore

RRF_K = 60


class HybridRetriever:
    def __init__(self, store: MemoryStore, *, rrf_k: int = RRF_K):
        self.store = store
        self.rrf_k = rrf_k

    # ------------------------------------------------------------------ public
    def search(
        self,
        query: str,
        limit: int = 10,
        scope: str | None = None,
        types: set[str] | list[str] | None = None,
        min_conf: float = 0.1,
    ) -> list[dict]:
        """Hybrid search. Returns a list of result dicts (id/content/type/
        scope/source/confidence/score), most relevant first."""
        if not query or not query.strip():
            return []
        bm25 = self.store.bm25_search(query, limit=limit * 3, scope=scope)
        vector: list[tuple[str, float]] = []
        qvec = self.store.embed_query(query)
        if qvec is not None:
            vector = self.store.vector_search(qvec, limit=limit * 3, scope=scope)

        fused = self._rrf_fuse(bm25, vector) if vector else bm25
        return self._apply_filters(fused, limit, scope, types, min_conf)

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
                }
            )
            if len(kept) >= limit:
                break
        # searching a memory counts as an access (feeds maintenance decay)
        self.store.mark_accessed([r["id"] for r in kept])
        return kept
