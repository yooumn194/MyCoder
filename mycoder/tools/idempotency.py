"""Cross-invocation tool idempotency.

A non-idempotent failure is the classic Agent footgun: a write tool fails
*after* applying its side effect, the retry re-runs it, and the effect lands
twice. Two complementary guards (see Agent._exec_tool / tools/base.py):

  1. IdempotencyStore — for idempotent tools, an identical `(tool_name, args)`
     call that already completed is served from cache instead of re-executed,
     so the agent cannot re-apply the same write twice by re-issuing it.
  2. retry_safe — non-idempotent tools are never auto-retried by
     run_with_correction (their side effect may already have happened).

The args fingerprint is a stable hash of the canonicalized arguments, so
argument order or non-JSON values don't defeat the key.
"""

from __future__ import annotations

import hashlib
import json
import threading
from typing import Any


def _fingerprint(args: dict[str, Any]) -> str:
    canonical = json.dumps(args, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


class IdempotencyStore:
    """Process/Agent-scoped record of completed (tool, args) executions.

    The dict is guarded by a lock because the agent executes parallel tool
    calls on a thread pool (agent.py) — concurrent get/put must not race.
    Parallel duplicate execution is additionally deduped in
    Agent._exec_tools_parallel, which reuses the same (tool, fingerprint) key.
    """

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], str] = {}
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def key(self, tool_name: str, args: dict[str, Any]) -> tuple[str, str]:
        return (tool_name, _fingerprint(args or {}))

    def get(self, key: tuple[str, str]) -> str | None:
        with self._lock:
            value = self._store.get(key)
            if value is not None:
                self._hits += 1
            else:
                self._misses += 1
            return value

    def put(self, key: tuple[str, str], result: str) -> None:
        with self._lock:
            self._store[key] = result

    def __contains__(self, key: tuple[str, str]) -> bool:
        with self._lock:
            return key in self._store

    def stats(self) -> dict:
        """Cache hit/miss counters — how often the idempotency cache saved a
        re-execution (面试「缓存命中率」的实测来源)."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "hits": self._hits,
                "misses": self._misses,
                "total": total,
                "hit_rate": round(self._hits / total, 4) if total else 0.0,
            }

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0
