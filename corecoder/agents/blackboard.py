"""Blackboard: lightweight shared-memory KV store for subagent communication.

Key convention: {task_id}:{artifact_type}:{name}. Data auto-expires after a
TTL. Concurrency is guarded by an asyncio.Lock (v1.0.1 fix — a threading.Lock
would deadlock inside an event loop). Subscribers fire on prefix matches.
"""

import asyncio
import time
from typing import Any, Callable, Optional


class Blackboard:
    def __init__(self, ttl_seconds: int = 300) -> None:
        self._store: dict[str, dict] = {}
        self._ttl = ttl_seconds
        self._lock = asyncio.Lock()
        self._subscribers: dict[str, list[Callable]] = {}

    async def put(self, task_id: str, key: str, value: Any, ttl: int | None = None) -> None:
        """Write a value with automatic expiry."""
        full_key = f"{task_id}:{key}"
        async with self._lock:
            self._store[full_key] = {
                "value": value,
                "timestamp": time.time(),
                "ttl": ttl if ttl is not None else self._ttl,
            }
        await self._notify(full_key, value)

    async def get(self, task_id: str, key: str) -> Optional[Any]:
        """Read a value, honoring expiry."""
        full_key = f"{task_id}:{key}"
        async with self._lock:
            entry = self._store.get(full_key)
            if entry is None:
                return None
            if time.time() - entry["timestamp"] > entry["ttl"]:
                del self._store[full_key]
                return None
            return entry["value"]

    async def query(self, task_id: str, prefix: str) -> dict[str, Any]:
        """Read all keys under {task_id}:{prefix}."""
        result: dict[str, Any] = {}
        async with self._lock:
            for full_key, entry in self._store.items():
                if full_key.startswith(f"{task_id}:{prefix}"):
                    result[full_key] = entry["value"]
        return result

    async def subscribe(self, pattern: str, callback: Callable) -> None:
        """Subscribe to writes whose key starts with `pattern`."""
        self._subscribers.setdefault(pattern, []).append(callback)

    async def _notify(self, key: str, value: Any) -> None:
        for pattern, callbacks in self._subscribers.items():
            if key.startswith(pattern):
                for cb in callbacks:
                    await cb(key, value)
