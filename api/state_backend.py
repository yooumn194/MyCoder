"""State backends for the MyCoder service layer.

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
project-root `.mycoder/` runtime dir) but keeps two dedicated tables.

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

_REDIS_SESSION_KEY = "mycoder:session:{id}"
_REDIS_BLACKBOARD_KEY = "mycoder:blackboard:{id}"
_REDIS_JOB_DATA = "mycoder:jobs:data"
_REDIS_JOB_PENDING = "mycoder:jobs:pending"
_REDIS_JOB_LEASED = "mycoder:jobs:leased"
_REDIS_JOB_OWNER = "mycoder:jobs:owner"
_REDIS_JOB_ATTEMPTS = "mycoder:jobs:attempts"
_REDIS_JOB_DEAD = "mycoder:jobs:dead"
_REDIS_LEASE_KEY = "mycoder:lease:{name}"
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

    @abstractmethod
    async def list_sessions(self) -> list[dict]:
        """Return every session record as {session_id, **data} — the source for
        production run success-rate aggregation (P2)."""

    @abstractmethod
    async def enqueue_job(self, job_id: str, payload: dict) -> bool:
        """Atomically enqueue a job; False means the id is already active."""

    @abstractmethod
    async def claim_job(
        self, worker_id: str, lease_seconds: int, job_id: str | None = None
    ) -> dict | None:
        """Atomically claim one available job, recovering expired claims."""

    @abstractmethod
    async def renew_job(self, job_id: str, worker_id: str, lease_seconds: int) -> bool:
        """Extend a running job's ownership lease."""

    @abstractmethod
    async def complete_job(self, job_id: str, worker_id: str) -> bool:
        """Acknowledge and remove a job owned by worker_id."""

    @abstractmethod
    async def retry_job(
        self, job_id: str, worker_id: str, delay_seconds: float = 0
    ) -> bool:
        """Return an owned job to the durable pending queue."""

    @abstractmethod
    async def dead_letter_job(
        self, job_id: str, worker_id: str, reason: str
    ) -> bool:
        """Atomically move an owned running job into the dead-letter store."""

    @abstractmethod
    async def list_dead_jobs(self, limit: int = 100) -> list[dict]:
        """Return newest dead-letter records for operations and debugging."""

    @abstractmethod
    async def acquire_lease(self, name: str, owner: str, lease_seconds: int) -> bool:
        """Acquire a named distributed lease."""

    @abstractmethod
    async def renew_lease(self, name: str, owner: str, lease_seconds: int) -> bool:
        """Extend a named lease only when owner still owns it."""

    @abstractmethod
    async def release_lease(self, name: str, owner: str) -> bool:
        """Release a named lease only when owner still owns it."""


def _dump(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False, default=str)


class LocalStateBackend(StateBackend):
    """SQLite-backed backend for local development (the default)."""

    def __init__(self, project_dir: str | Path | None = None) -> None:
        root = Path(project_dir or os.getcwd()).expanduser().resolve()
        self.path = root / ".mycoder" / _STATE_DB_NAME
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
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    status TEXT NOT NULL,
                    owner TEXT,
                    lease_until REAL,
                    available_at REAL NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_jobs_claim
                    ON jobs(status, available_at, updated_at);
                CREATE TABLE IF NOT EXISTS dead_jobs (
                    job_id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    attempts INTEGER NOT NULL,
                    reason TEXT NOT NULL,
                    failed_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_dead_jobs_failed_at ON dead_jobs(failed_at);
                CREATE TABLE IF NOT EXISTS leases (
                    name TEXT PRIMARY KEY,
                    owner TEXT NOT NULL,
                    expires_at REAL NOT NULL
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

    async def list_sessions(self) -> list[dict]:
        conn = self._connect()
        rows = conn.execute("SELECT session_id, data FROM sessions").fetchall()
        return [
            {"session_id": r["session_id"], **json.loads(r["data"])}
            for r in rows
        ]

    async def enqueue_job(self, job_id: str, payload: dict) -> bool:
        now = time.time()
        cur = self._connect().execute(
            "INSERT OR IGNORE INTO jobs "
            "(job_id, data, status, owner, lease_until, available_at, attempts, updated_at) "
            "VALUES (?, ?, 'queued', NULL, NULL, ?, 0, ?)",
            (job_id, _dump(payload), now, now),
        )
        self._connect().commit()
        return cur.rowcount == 1

    async def claim_job(
        self, worker_id: str, lease_seconds: int, job_id: str | None = None
    ) -> dict | None:
        conn = self._connect()
        now = time.time()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "UPDATE jobs SET status='queued', owner=NULL, lease_until=NULL, "
                "available_at=?, updated_at=? "
                "WHERE status='running' AND lease_until <= ?",
                (now, now, now),
            )
            if job_id is None:
                row = conn.execute(
                    "SELECT job_id, data, attempts FROM jobs "
                    "WHERE status='queued' AND available_at <= ? "
                    "ORDER BY available_at, updated_at LIMIT 1",
                    (now,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT job_id, data, attempts FROM jobs "
                    "WHERE job_id=? AND status='queued' AND available_at <= ?",
                    (job_id, now),
                ).fetchone()
            if row is None:
                conn.commit()
                return None
            changed = conn.execute(
                "UPDATE jobs SET status='running', owner=?, lease_until=?, "
                "attempts=attempts+1, updated_at=? "
                "WHERE job_id=? AND status='queued'",
                (worker_id, now + max(1, lease_seconds), now, row["job_id"]),
            ).rowcount
            conn.commit()
            if changed != 1:
                return None
            return {
                "job_id": row["job_id"],
                "payload": json.loads(row["data"]),
                "attempts": int(row["attempts"]) + 1,
            }
        except Exception:
            conn.rollback()
            raise

    async def renew_job(self, job_id: str, worker_id: str, lease_seconds: int) -> bool:
        now = time.time()
        cur = self._connect().execute(
            "UPDATE jobs SET lease_until=?, updated_at=? "
            "WHERE job_id=? AND status='running' AND owner=?",
            (now + max(1, lease_seconds), now, job_id, worker_id),
        )
        self._connect().commit()
        return cur.rowcount == 1

    async def complete_job(self, job_id: str, worker_id: str) -> bool:
        cur = self._connect().execute(
            "DELETE FROM jobs WHERE job_id=? AND status='running' AND owner=?",
            (job_id, worker_id),
        )
        self._connect().commit()
        return cur.rowcount == 1

    async def retry_job(
        self, job_id: str, worker_id: str, delay_seconds: float = 0
    ) -> bool:
        now = time.time()
        cur = self._connect().execute(
            "UPDATE jobs SET status='queued', owner=NULL, lease_until=NULL, "
            "available_at=?, updated_at=? "
            "WHERE job_id=? AND status='running' AND owner=?",
            (now + max(0, delay_seconds), now, job_id, worker_id),
        )
        self._connect().commit()
        return cur.rowcount == 1

    async def dead_letter_job(
        self, job_id: str, worker_id: str, reason: str
    ) -> bool:
        conn = self._connect()
        now = time.time()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT data, attempts FROM jobs "
                "WHERE job_id=? AND status='running' AND owner=?",
                (job_id, worker_id),
            ).fetchone()
            if row is None:
                conn.commit()
                return False
            conn.execute(
                "INSERT INTO dead_jobs(job_id, data, attempts, reason, failed_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(job_id) DO UPDATE SET data=excluded.data, "
                "attempts=excluded.attempts, reason=excluded.reason, "
                "failed_at=excluded.failed_at",
                (job_id, row["data"], int(row["attempts"]), reason[:2000], now),
            )
            conn.execute(
                "DELETE FROM jobs WHERE job_id=? AND status='running' AND owner=?",
                (job_id, worker_id),
            )
            conn.commit()
            return True
        except Exception:
            conn.rollback()
            raise

    async def list_dead_jobs(self, limit: int = 100) -> list[dict]:
        rows = self._connect().execute(
            "SELECT job_id, data, attempts, reason, failed_at FROM dead_jobs "
            "ORDER BY failed_at DESC LIMIT ?",
            (max(1, min(int(limit), 1000)),),
        ).fetchall()
        return [
            {
                "job_id": row["job_id"],
                "payload": json.loads(row["data"]),
                "attempts": int(row["attempts"]),
                "reason": row["reason"],
                "failed_at": float(row["failed_at"]),
            }
            for row in rows
        ]

    async def acquire_lease(self, name: str, owner: str, lease_seconds: int) -> bool:
        conn = self._connect()
        now = time.time()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("DELETE FROM leases WHERE name=? AND expires_at <= ?", (name, now))
            changed = conn.execute(
                "INSERT OR IGNORE INTO leases(name, owner, expires_at) VALUES (?, ?, ?)",
                (name, owner, now + max(1, lease_seconds)),
            ).rowcount
            conn.commit()
            return changed == 1
        except Exception:
            conn.rollback()
            raise

    async def renew_lease(self, name: str, owner: str, lease_seconds: int) -> bool:
        now = time.time()
        cur = self._connect().execute(
            "UPDATE leases SET expires_at=? WHERE name=? AND owner=? AND expires_at > ?",
            (now + max(1, lease_seconds), name, owner, now),
        )
        self._connect().commit()
        return cur.rowcount == 1

    async def release_lease(self, name: str, owner: str) -> bool:
        cur = self._connect().execute(
            "DELETE FROM leases WHERE name=? AND owner=?", (name, owner)
        )
        self._connect().commit()
        return cur.rowcount == 1


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

    async def list_sessions(self) -> list[dict]:
        out: list[dict] = []
        pattern = _REDIS_SESSION_KEY.format(id="*")
        async for key in self._redis.scan_iter(match=pattern):
            raw = await self._redis.get(key)
            if raw:
                sid = str(key).rsplit(":", 1)[-1]
                try:
                    out.append({"session_id": sid, **json.loads(raw)})
                except (json.JSONDecodeError, TypeError):
                    continue
        return out

    async def enqueue_job(self, job_id: str, payload: dict) -> bool:
        script = """
        if redis.call('HEXISTS', KEYS[1], ARGV[1]) == 1 then return 0 end
        redis.call('HSET', KEYS[1], ARGV[1], ARGV[2])
        redis.call('ZADD', KEYS[2], ARGV[3], ARGV[1])
        redis.call('HSET', KEYS[3], ARGV[1], 0)
        return 1
        """
        result = await self._redis.eval(
            script, 3, _REDIS_JOB_DATA, _REDIS_JOB_PENDING,
            _REDIS_JOB_ATTEMPTS, job_id, _dump(payload), time.time(),
        )
        return bool(result)

    async def claim_job(
        self, worker_id: str, lease_seconds: int, job_id: str | None = None
    ) -> dict | None:
        script = """
        local expired = redis.call('ZRANGEBYSCORE', KEYS[2], '-inf', ARGV[1])
        for _, id in ipairs(expired) do
          redis.call('ZREM', KEYS[2], id)
          redis.call('HDEL', KEYS[4], id)
          redis.call('ZADD', KEYS[1], ARGV[1], id)
        end
        local id = ARGV[4]
        if id == '' then
          local ready = redis.call('ZRANGEBYSCORE', KEYS[1], '-inf', ARGV[1], 'LIMIT', 0, 1)
          if #ready == 0 then return nil end
          id = ready[1]
        elseif redis.call('ZSCORE', KEYS[1], id) == false
          or tonumber(redis.call('ZSCORE', KEYS[1], id)) > tonumber(ARGV[1]) then
          return nil
        end
        if redis.call('ZREM', KEYS[1], id) ~= 1 then return nil end
        redis.call('ZADD', KEYS[2], tonumber(ARGV[1]) + tonumber(ARGV[2]), id)
        redis.call('HSET', KEYS[4], id, ARGV[3])
        local attempts = redis.call('HINCRBY', KEYS[5], id, 1)
        local data = redis.call('HGET', KEYS[3], id)
        if data == false then return nil end
        return {id, data, tostring(attempts)}
        """
        raw = await self._redis.eval(
            script, 5, _REDIS_JOB_PENDING, _REDIS_JOB_LEASED,
            _REDIS_JOB_DATA, _REDIS_JOB_OWNER, _REDIS_JOB_ATTEMPTS,
            time.time(), max(1, lease_seconds), worker_id, job_id or "",
        )
        if not raw:
            return None
        return {"job_id": raw[0], "payload": json.loads(raw[1]), "attempts": int(raw[2])}

    async def renew_job(self, job_id: str, worker_id: str, lease_seconds: int) -> bool:
        script = """
        if redis.call('HGET', KEYS[2], ARGV[1]) ~= ARGV[2] then return 0 end
        if redis.call('ZSCORE', KEYS[1], ARGV[1]) == false then return 0 end
        redis.call('ZADD', KEYS[1], tonumber(ARGV[3]) + tonumber(ARGV[4]), ARGV[1])
        return 1
        """
        result = await self._redis.eval(
            script, 2, _REDIS_JOB_LEASED, _REDIS_JOB_OWNER,
            job_id, worker_id, time.time(), max(1, lease_seconds),
        )
        return bool(result)

    async def complete_job(self, job_id: str, worker_id: str) -> bool:
        script = """
        if redis.call('HGET', KEYS[2], ARGV[1]) ~= ARGV[2] then return 0 end
        redis.call('ZREM', KEYS[1], ARGV[1])
        redis.call('HDEL', KEYS[2], ARGV[1])
        redis.call('HDEL', KEYS[3], ARGV[1])
        redis.call('HDEL', KEYS[4], ARGV[1])
        return 1
        """
        result = await self._redis.eval(
            script, 4, _REDIS_JOB_LEASED, _REDIS_JOB_OWNER,
            _REDIS_JOB_DATA, _REDIS_JOB_ATTEMPTS, job_id, worker_id,
        )
        return bool(result)

    async def retry_job(
        self, job_id: str, worker_id: str, delay_seconds: float = 0
    ) -> bool:
        script = """
        if redis.call('HGET', KEYS[3], ARGV[1]) ~= ARGV[2] then return 0 end
        redis.call('ZREM', KEYS[2], ARGV[1])
        redis.call('HDEL', KEYS[3], ARGV[1])
        redis.call('ZADD', KEYS[1], tonumber(ARGV[3]) + tonumber(ARGV[4]), ARGV[1])
        return 1
        """
        result = await self._redis.eval(
            script, 3, _REDIS_JOB_PENDING, _REDIS_JOB_LEASED,
            _REDIS_JOB_OWNER, job_id, worker_id, time.time(), max(0, delay_seconds),
        )
        return bool(result)

    async def dead_letter_job(
        self, job_id: str, worker_id: str, reason: str
    ) -> bool:
        script = """
        if redis.call('HGET', KEYS[2], ARGV[1]) ~= ARGV[2] then return 0 end
        local data = redis.call('HGET', KEYS[3], ARGV[1])
        if data == false then return 0 end
        local attempts = tonumber(redis.call('HGET', KEYS[4], ARGV[1]) or '0')
        local ok, payload = pcall(cjson.decode, data)
        if not ok then payload = {raw = data} end
        local record = cjson.encode({
          job_id = ARGV[1], payload = payload, attempts = attempts,
          reason = ARGV[3], failed_at = tonumber(ARGV[4])
        })
        redis.call('HSET', KEYS[5], ARGV[1], record)
        redis.call('ZREM', KEYS[1], ARGV[1])
        redis.call('HDEL', KEYS[2], ARGV[1])
        redis.call('HDEL', KEYS[3], ARGV[1])
        redis.call('HDEL', KEYS[4], ARGV[1])
        return 1
        """
        result = await self._redis.eval(
            script,
            5,
            _REDIS_JOB_LEASED,
            _REDIS_JOB_OWNER,
            _REDIS_JOB_DATA,
            _REDIS_JOB_ATTEMPTS,
            _REDIS_JOB_DEAD,
            job_id,
            worker_id,
            reason[:2000],
            time.time(),
        )
        return bool(result)

    async def list_dead_jobs(self, limit: int = 100) -> list[dict]:
        raw_records = await self._redis.hvals(_REDIS_JOB_DEAD)
        records: list[dict] = []
        for raw in raw_records:
            try:
                record = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(record, dict):
                records.append(record)
        records.sort(key=lambda item: float(item.get("failed_at", 0)), reverse=True)
        return records[: max(1, min(int(limit), 1000))]

    async def acquire_lease(self, name: str, owner: str, lease_seconds: int) -> bool:
        result = await self._redis.set(
            _REDIS_LEASE_KEY.format(name=name), owner, nx=True, ex=max(1, lease_seconds)
        )
        return bool(result)

    async def renew_lease(self, name: str, owner: str, lease_seconds: int) -> bool:
        script = """
        if redis.call('GET', KEYS[1]) ~= ARGV[1] then return 0 end
        return redis.call('EXPIRE', KEYS[1], ARGV[2])
        """
        result = await self._redis.eval(
            script, 1, _REDIS_LEASE_KEY.format(name=name), owner, max(1, lease_seconds)
        )
        return bool(result)

    async def release_lease(self, name: str, owner: str) -> bool:
        script = """
        if redis.call('GET', KEYS[1]) ~= ARGV[1] then return 0 end
        return redis.call('DEL', KEYS[1])
        """
        result = await self._redis.eval(
            script, 1, _REDIS_LEASE_KEY.format(name=name), owner
        )
        return bool(result)

    async def ping(self) -> bool:
        """Used by /health; False on any connection error."""
        try:
            return bool(await self._redis.ping())
        except Exception:  # noqa: BLE001 - health probe must never raise
            return False

    async def close(self) -> None:
        """Close the Redis connection on process shutdown (#14)."""
        try:
            await self._redis.aclose()
        except Exception:  # noqa: BLE001 - best-effort teardown
            pass


def create_state_backend(backend_type: str | None = None) -> StateBackend:
    """Factory: STATE_BACKEND=redis|local (default local). An unknown value
    falls back to local so a bad env var never bricks the service."""
    backend_type = (backend_type or os.getenv("STATE_BACKEND", "local")).strip().lower()
    if backend_type == "redis":
        return RedisStateBackend()
    return LocalStateBackend()
