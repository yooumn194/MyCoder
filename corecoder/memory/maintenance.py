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
