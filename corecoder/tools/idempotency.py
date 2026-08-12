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
from typing import Any


def _fingerprint(args: dict[str, Any]) -> str:
    canonical = json.dumps(args, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


class IdempotencyStore:
    """Process/Agent-scoped record of completed (tool, args) executions."""

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], str] = {}

    def key(self, tool_name: str, args: dict[str, Any]) -> tuple[str, str]:
        return (tool_name, _fingerprint(args or {}))

    def get(self, key: tuple[str, str]) -> str | None:
        return self._store.get(key)

    def put(self, key: tuple[str, str], result: str) -> None:
        self._store[key] = result

    def __contains__(self, key: tuple[str, str]) -> bool:
        return key in self._store

    def clear(self) -> None:
        self._store.clear()
