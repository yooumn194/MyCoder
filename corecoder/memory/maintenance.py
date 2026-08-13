"""MemoryMaintainer: confidence decay, compaction and statistics.

Decay rules (per spec):
  * a memory decays when it is source='auto', was never confirmed by the user,
    has been accessed at least 3 times, and was last accessed more than
    decay_days ago;
  * project memories decay faster (factor 0.8), global memories slower (0.95);
  * user / confirmed memories NEVER decay;
  * when confidence drops below the threshold the memory is flagged
    deprecated_by='decayed' (hidden from search until compact() prunes it).
"""

from __future__ import annotations

import time

from .store import MemoryStore
from .tokenizer import tokenize_chinese
from .types import DECAY_THRESHOLD

_DECAY_FACTOR = {"project": 0.8, "global": 0.95}


class MemoryMaintainer:
    def __init__(
        self,
        store: MemoryStore,
        *,
        decay_days: int = 30,
        confidence_threshold: float = DECAY_THRESHOLD,
    ):
        self.store = store
        self.decay_days = decay_days
        self.confidence_threshold = confidence_threshold

    # ------------------------------------------------------------------ decay
    def decay(self) -> int:
        """Apply one decay pass. Returns the number of memories decayed."""
        now = time.time()
        cutoff = now - self.decay_days * 86400
        decayed = 0
        for db in self.store._dbs():  # noqa: SLF001 - same-package store access
            rows = db.conn.execute(
                "SELECT id, confidence, scope FROM memories "
                "WHERE source = 'auto' AND access_count >= 3 "
                "AND last_accessed < ? AND deprecated_by IS NULL",
                (cutoff,),
            ).fetchall()
            for row in rows:
                factor = _DECAY_FACTOR.get(row["scope"], 0.8)
                new_conf = round(float(row["confidence"]) * factor, 4)
                deprecated = "decayed" if new_conf < self.confidence_threshold else None
                with db.conn:
                    db.conn.execute(
                        "UPDATE memories SET confidence = ?, deprecated_by = ?, "
                        "updated_at = ? WHERE id = ?",
                        (new_conf, deprecated, now, row["id"]),
                    )
                decayed += 1
        return decayed

    # ---------------------------------------------------------------- compact
    def compact(self) -> int:
        """Prune deprecated / below-threshold memories.

        SELECT-then-DELETE (not DELETE WHERE subquery) keeps compatibility with
        SQLite < 3.35, where a DELETE with a subquery against the same table
        can't reuse indexes. Returns the number of rows removed.
        """
        removed = 0
        for db in self.store._dbs():  # noqa: SLF001
            rows = db.conn.execute(
                "SELECT id FROM memories WHERE deprecated_by IS NOT NULL "
                "OR confidence < ?",
                (self.confidence_threshold,),
            ).fetchall()
            for row in rows:
                if self.store.delete(row["id"]):
                    removed += 1
        return removed

    # ------------------------------------------------------------ integrity
    def audit_integrity(
        self,
        store: MemoryStore | None = None,
        *,
        min_overlap: float = 0.5,
        min_shared_terms: int = 2,
        low_confidence_threshold: float = 0.3,
    ) -> list[dict]:
        """Scan active memories for pollution (P1, wrong-memory correction).

        Flags two failure modes:
          * low_confidence — an `auto` memory that is still active but whose
            confidence already dropped (its reliability is doubtful);
          * conflicting    — two active memories on the SAME topic (high token
            overlap) that are NOT near-duplicates: their different answers
            can't both be right.

        Returns a deduped list of issue dicts:
          {id, content, confidence, issue, related_id?, related_content?}
        """
        store = store or self.store
        active = store.list(include_deprecated=False)
        flagged: list[dict] = []

        for e in active:
            if e.source == "auto" and e.confidence < low_confidence_threshold:
                flagged.append(
                    {
                        "id": e.id,
                        "content": e.content[:120],
                        "confidence": e.confidence,
                        "issue": "low_confidence",
                    }
                )

        tokens = [(e.id, set(tokenize_chinese(e.content).split()), e) for e in active]
        for i in range(len(tokens)):
            for j in range(i + 1, len(tokens)):
                id_a, ta, ea = tokens[i]
                id_b, tb, eb = tokens[j]
                shared = ta & tb
                if len(shared) < min_shared_terms:
                    continue
                union = ta | tb
                overlap = len(shared) / len(union) if union else 0.0
                if overlap >= min_overlap:
                    flagged.append(
                        {
                            "id": id_a,
                            "content": ea.content[:120],
                            "confidence": ea.confidence,
                            "issue": "conflicting",
                            "related_id": id_b,
                            "related_content": eb.content[:120],
                        }
                    )
                    flagged.append(
                        {
                            "id": id_b,
                            "content": eb.content[:120],
                            "confidence": eb.confidence,
                            "issue": "conflicting",
                            "related_id": id_a,
                            "related_content": ea.content[:120],
                        }
                    )

        # dedupe by id, keep the first issue recorded
        seen: dict[str, dict] = {}
        for f in flagged:
            seen.setdefault(f["id"], f)
        return list(seen.values())

    def correct_memory(
        self,
        mem_id: str,
        *,
        content: str | None = None,
        reason: str = "corrected",
        store: MemoryStore | None = None,
    ) -> bool:
        """Fix a polluted memory (P1). Returns True when the memory existed.

          * content given -> replace the wrong content, clear the deprecation
            and raise confidence (the corrected fact becomes usable again);
          * content None  -> deprecate the memory so it stops surfacing until a
            human/agent rewrites it.
        """
        store = store or self.store
        entry = store.get(mem_id)
        if entry is None:
            return False
        if content is not None:
            store.update(
                mem_id,
                content=content,
                deprecated_by=None,
                confidence=max(float(entry.confidence or 0.5), 0.6),
            )
        else:
            store.update(mem_id, deprecated_by=reason)
        return True

    # ------------------------------------------------------------------ stats
    def get_stats(self) -> dict:
        """Aggregate counters across both dbs."""
        total = active = 0
        avg_numerator = 0.0
        by_type: dict[str, int] = {}
        by_scope: dict[str, int] = {}
        for db in self.store._dbs():  # noqa: SLF001
            row = db.conn.execute(
                "SELECT COUNT(*) AS n, "
                "SUM(CASE WHEN deprecated_by IS NULL THEN 1 ELSE 0 END) AS active, "
                "COALESCE(AVG(confidence), 0.0) AS avg_conf "
                "FROM memories"
            ).fetchone()
            total += int(row["n"] or 0)
            active += int(row["active"] or 0)
            avg_numerator += float(row["avg_conf"]) * int(row["n"] or 0)
            for trow in db.conn.execute(
                "SELECT type, COUNT(*) AS n FROM memories GROUP BY type"
            ):
                by_type[trow["type"]] = by_type.get(trow["type"], 0) + int(trow["n"])
            for srow in db.conn.execute(
                "SELECT scope, COUNT(*) AS n FROM memories GROUP BY scope"
            ):
                by_scope[srow["scope"]] = by_scope.get(srow["scope"], 0) + int(srow["n"])
        return {
            "total": total,
            "active": active,
            "deprecated": total - active,
            "by_type": dict(sorted(by_type.items())),
            "by_scope": dict(sorted(by_scope.items())),
            "avg_confidence": round(avg_numerator / total, 4) if total else 0.0,
        }
