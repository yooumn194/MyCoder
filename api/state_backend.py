"""State backends for the CoreCoder service layer.

Sessions and blackboards must survive the stateless HTTP boundary, so every
API request reads/writes through a StateBackend. Two implementations:

  * LocalStateBackend  — stdlib sqlite3, the default (`STATE_BACKEND` unset).
  * RedisStateBackend  — redis.asyncio, namespaced keys + 24h TTL.

Why not store sessions in MemoryStore's `memories` table: MemoryStore feeds
the hybrid-retrieval index — `memories` is CHECK-constrained to memory types
and every row is tokenized, embedded, deduplicated and decayed. Session and
blackboard blobs are not memories; putting them there would pollute
`memory_search`/`memory_list` and confuse confidence decay. So LocalStateBackend
reuses the *same* SQLite approach (stdlib sqlite3, lazy connection, row_factory,
project-root `.corecoder/` runtime dir) but keeps two dedicated tables.

Redis is imported lazily inside the constructor: a plain install without the
`redis` client keeps local mode working, and `STATE_BACKEND=redis` fails fast
with a clear ImportError instead of crashing at import time.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from abc import ABC, abstractmethod
from pathlib import Path

_REDIS_SESSION_KEY = "corecoder:session:{id}"
_REDIS_BLACKBOARD_KEY = "corecoder:blackboard:{id}"
_REDIS_TTL_SECONDS = 24 * 60 * 60
_STATE_DB_NAME = "api_state.db"
_DEFAULT_REDIS_URL = "redis://localhost:6379/0"


class StateBackend(ABC):
    """Persistence contract for one session's mutable state."""

    @abstractmethod
    async def get_session(self, session_id: str) -> dict | None:
        """Return the session record, or None when the session is unknown."""

    @abstractmethod
    async def save_session(self, session_id: str, data: dict) -> None:
        """Upsert the session record."""

    @abstractmethod
    async def get_blackboard(self, session_id: str) -> dict | None:
        """Return the blackboard snapshot, or None."""

    @abstractmethod
    async def save_blackboard(self, session_id: str, data: dict) -> None:
        """Persist the blackboard snapshot."""


def _dump(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False, default=str)


class LocalStateBackend(StateBackend):
    """SQLite-backed backend for local development (the default)."""

    def __init__(self, project_dir: str | Path | None = None) -> None:
        root = Path(project_dir or os.getcwd()).expanduser().resolve()
        self.path = root / ".corecoder" / _STATE_DB_NAME
        self._conn: sqlite3.Connection | None = None
        self._tables_ready = False

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.path))
            self._conn.row_factory = sqlite3.Row
        if not self._tables_ready:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS blackboards (
                    session_id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
                """
            )
            self._conn.commit()
            self._tables_ready = True
        return self._conn

    async def get_session(self, session_id: str) -> dict | None:
        row = self._connect().execute(
            "SELECT data FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        return json.loads(row["data"]) if row else None

    async def save_session(self, session_id: str, data: dict) -> None:
        conn = self._connect()
        conn.execute(
            "INSERT INTO sessions (session_id, data, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(session_id) DO UPDATE SET data=excluded.data, "
            "updated_at=excluded.updated_at",
            (session_id, _dump(data), time.time()),
        )
        conn.commit()

    async def get_blackboard(self, session_id: str) -> dict | None:
        row = self._connect().execute(
            "SELECT data FROM blackboards WHERE session_id = ?", (session_id,)
        ).fetchone()
        return json.loads(row["data"]) if row else None

    async def save_blackboard(self, session_id: str, data: dict) -> None:
        conn = self._connect()
        conn.execute(
            "INSERT INTO blackboards (session_id, data, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(session_id) DO UPDATE SET data=excluded.data, "
            "updated_at=excluded.updated_at",
            (session_id, _dump(data), time.time()),
        )
        conn.commit()


class RedisStateBackend(StateBackend):
    """Redis-backed backend (STATE_BACKEND=redis). Keys are namespaced so they
    can never collide with another app's data in a shared Redis."""

    def __init__(self, url: str | None = None, ttl: int = _REDIS_TTL_SECONDS) -> None:
        import redis.asyncio as aioredis  # noqa: F401 - ImportError = redis not installed

        self._redis = aioredis.from_url(
            url or os.getenv("REDIS_URL", _DEFAULT_REDIS_URL), decode_responses=True
        )
        self.ttl = ttl

    async def _get(self, key: str) -> dict | None:
        raw = await self._redis.get(key)
        return json.loads(raw) if raw else None

    async def get_session(self, session_id: str) -> dict | None:
        return await self._get(_REDIS_SESSION_KEY.format(id=session_id))

    async def save_session(self, session_id: str, data: dict) -> None:
        await self._redis.set(
            _REDIS_SESSION_KEY.format(id=session_id), _dump(data), ex=self.ttl
        )

    async def get_blackboard(self, session_id: str) -> dict | None:
        return await self._get(_REDIS_BLACKBOARD_KEY.format(id=session_id))

    async def save_blackboard(self, session_id: str, data: dict) -> None:
        await self._redis.set(
            _REDIS_BLACKBOARD_KEY.format(id=session_id), _dump(data), ex=self.ttl
        )

    async def ping(self) -> bool:
        """Used by /health; False on any connection error."""
        try:
            return bool(await self._redis.ping())
        except Exception:  # noqa: BLE001 - health probe must never raise
            return False


def create_state_backend(backend_type: str | None = None) -> StateBackend:
    """Factory: STATE_BACKEND=redis|local (default local). An unknown value
    falls back to local so a bad env var never bricks the service."""
    backend_type = (backend_type or os.getenv("STATE_BACKEND", "local")).strip().lower()
    if backend_type == "redis":
        return RedisStateBackend()
    return LocalStateBackend()
