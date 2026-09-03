"""Process-independent storage for API observability state.

LLM calls are traced from worker threads, so this store intentionally exposes
a synchronous contract. Local mode uses short-lived SQLite connections (safe
across threads/processes); Redis mode uses the synchronous Redis client and
atomic Lua for rate-limit admission and alert cooldown claims.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Protocol

_DEFAULT_TTL_SECONDS = 7 * 24 * 60 * 60
_DEFAULT_MAX_ALERTS = 10_000


class TraceStore(Protocol):
    def append_trace(self, trace: dict) -> None: ...

    def list_traces(self, session_id: str | None = None) -> list[dict]: ...

    def list_trace_sessions(self) -> list[str]: ...

    def reset_traces(self) -> None: ...


class RateLimitStore(Protocol):
    def rate_limit_allow(
        self, key: str, limit: int, window_seconds: float = 60.0
    ) -> bool: ...


class AlertStore(Protocol):
    def claim_alert(
        self,
        session_id: str,
        rule: str,
        cooldown_seconds: int,
        alert: dict,
    ) -> bool: ...

    def list_alerts(self, limit: int = 100) -> list[dict]: ...

    def reset_alerts(self) -> None: ...


class ObservabilityStore(TraceStore, RateLimitStore, AlertStore, Protocol):
    """Complete service store; consumers should type against a narrow protocol."""

    def close(self) -> None: ...


class SQLiteObservabilityStore:
    """SQLite observability state shared by local API workers/processes."""

    def __init__(
        self,
        path: str | Path,
        *,
        ttl_seconds: int = _DEFAULT_TTL_SECONDS,
        max_alerts: int = _DEFAULT_MAX_ALERTS,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = max(60, int(ttl_seconds))
        self.max_alerts = max(1, int(max_alerts))
        self._init_lock = threading.Lock()
        self._ready = False
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _ensure_schema(self) -> None:
        if self._ready:
            return
        with self._init_lock:
            if self._ready:
                return
            with self._connect() as conn:
                conn.executescript(
                    """
                    PRAGMA journal_mode=WAL;
                    CREATE TABLE IF NOT EXISTS observability_traces (
                        call_id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        data TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_observability_trace_session
                        ON observability_traces(session_id, timestamp);
                    CREATE TABLE IF NOT EXISTS observability_rate_hits (
                        hit_id TEXT PRIMARY KEY,
                        rate_key TEXT NOT NULL,
                        hit_at REAL NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_observability_rate_window
                        ON observability_rate_hits(rate_key, hit_at);
                    CREATE INDEX IF NOT EXISTS idx_observability_rate_time
                        ON observability_rate_hits(hit_at);
                    CREATE TABLE IF NOT EXISTS observability_alert_cooldowns (
                        session_id TEXT NOT NULL,
                        rule TEXT NOT NULL,
                        last_fired REAL NOT NULL,
                        PRIMARY KEY(session_id, rule)
                    );
                    CREATE TABLE IF NOT EXISTS observability_alerts (
                        alert_id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        rule TEXT NOT NULL,
                        fired_at REAL NOT NULL,
                        data TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_observability_alert_time
                        ON observability_alerts(fired_at DESC);
                    """
                )
            self._ready = True

    def append_trace(self, trace: dict) -> None:
        self._ensure_schema()
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO observability_traces
                   (call_id, session_id, timestamp, data) VALUES (?, ?, ?, ?)""",
                (
                    str(trace["call_id"]),
                    str(trace["session_id"]),
                    float(trace.get("timestamp", time.time())),
                    json.dumps(trace, ensure_ascii=False, default=str),
                ),
            )
            conn.execute(
                "DELETE FROM observability_traces WHERE timestamp < ?",
                (now - self.ttl_seconds,),
            )

    def list_traces(self, session_id: str | None = None) -> list[dict]:
        self._ensure_schema()
        query = "SELECT data FROM observability_traces"
        params: tuple[Any, ...] = ()
        if session_id is not None:
            query += " WHERE session_id = ?"
            params = (session_id,)
        query += " ORDER BY timestamp, call_id"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [json.loads(row["data"]) for row in rows]

    def list_trace_sessions(self) -> list[str]:
        self._ensure_schema()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT session_id FROM observability_traces ORDER BY session_id"
            ).fetchall()
        return [str(row["session_id"]) for row in rows]

    def reset_traces(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM observability_traces")

    def rate_limit_allow(
        self, key: str, limit: int, window_seconds: float = 60.0
    ) -> bool:
        if limit <= 0:
            return False
        now = time.time()
        cutoff = now - window_seconds
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("DELETE FROM observability_rate_hits WHERE hit_at < ?", (cutoff,))
            count = int(
                conn.execute(
                    "SELECT COUNT(*) FROM observability_rate_hits WHERE rate_key = ?",
                    (key,),
                ).fetchone()[0]
            )
            if count >= limit:
                # Keep the global stale-hit cleanup even when this key is denied.
                conn.commit()
                return False
            conn.execute(
                "INSERT INTO observability_rate_hits(hit_id, rate_key, hit_at) VALUES (?, ?, ?)",
                (uuid.uuid4().hex, key, now),
            )
            conn.commit()
            return True
        finally:
            conn.close()

    def claim_alert(
        self,
        session_id: str,
        rule: str,
        cooldown_seconds: int,
        alert: dict,
    ) -> bool:
        now = time.time()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """SELECT last_fired FROM observability_alert_cooldowns
                   WHERE session_id = ? AND rule = ?""",
                (session_id, rule),
            ).fetchone()
            if row is not None and now - float(row[0]) < max(0, cooldown_seconds):
                conn.rollback()
                return False
            conn.execute(
                """INSERT INTO observability_alert_cooldowns(session_id, rule, last_fired)
                   VALUES (?, ?, ?)
                   ON CONFLICT(session_id, rule) DO UPDATE SET last_fired=excluded.last_fired""",
                (session_id, rule, now),
            )
            record = {**alert, "fired_at": now}
            conn.execute(
                """INSERT INTO observability_alerts
                   (alert_id, session_id, rule, fired_at, data) VALUES (?, ?, ?, ?, ?)""",
                (
                    uuid.uuid4().hex,
                    session_id,
                    rule,
                    now,
                    json.dumps(record, ensure_ascii=False, default=str),
                ),
            )
            cutoff = now - self.ttl_seconds
            conn.execute(
                "DELETE FROM observability_alert_cooldowns WHERE last_fired < ?",
                (cutoff,),
            )
            conn.execute(
                "DELETE FROM observability_alerts WHERE fired_at < ?", (cutoff,)
            )
            conn.execute(
                """DELETE FROM observability_alerts WHERE alert_id NOT IN (
                       SELECT alert_id FROM observability_alerts
                       ORDER BY fired_at DESC LIMIT ?
                   )""",
                (self.max_alerts,),
            )
            conn.commit()
            return True
        finally:
            conn.close()

    def list_alerts(self, limit: int = 100) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT data FROM observability_alerts ORDER BY fired_at DESC LIMIT ?",
                (max(1, min(int(limit), 1000)),),
            ).fetchall()
        return [json.loads(row["data"]) for row in rows]

    def reset_alerts(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM observability_alert_cooldowns")
            conn.execute("DELETE FROM observability_alerts")

    def close(self) -> None:
        # Connections are deliberately per-operation for cross-thread/process safety.
        return None


class RedisObservabilityStore:
    """Redis observability state shared by every API worker/replica."""

    _RATE_LIMIT_SCRIPT = """
    local now = redis.call('TIME')
    local now_ms = (now[1] * 1000) + math.floor(now[2] / 1000)
    local window_ms = tonumber(ARGV[1])
    redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now_ms - window_ms)
    if redis.call('ZCARD', KEYS[1]) >= tonumber(ARGV[2]) then return 0 end
    redis.call('ZADD', KEYS[1], now_ms, ARGV[3])
    redis.call('PEXPIRE', KEYS[1], window_ms)
    return 1
    """
    _ALERT_SCRIPT = """
    if not redis.call('SET', KEYS[1], '1', 'PX', ARGV[1], 'NX') then return 0 end
    redis.call('LPUSH', KEYS[2], ARGV[2])
    redis.call('LTRIM', KEYS[2], 0, tonumber(ARGV[3]) - 1)
    redis.call('EXPIRE', KEYS[2], tonumber(ARGV[4]))
    return 1
    """

    def __init__(
        self,
        url: str,
        *,
        ttl_seconds: int = _DEFAULT_TTL_SECONDS,
        max_alerts: int = _DEFAULT_MAX_ALERTS,
    ) -> None:
        import redis

        self._redis = redis.Redis.from_url(url, decode_responses=True)
        self.ttl_seconds = max(60, int(ttl_seconds))
        self.max_alerts = max(1, int(max_alerts))
        self._prefix = "mycoder:observability"

    @staticmethod
    def _digest(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _trace_key(self, session_id: str) -> str:
        return f"{self._prefix}:traces:{self._digest(session_id)}"

    def append_trace(self, trace: dict) -> None:
        session_id = str(trace["session_id"])
        timestamp = float(trace.get("timestamp", time.time()))
        key = self._trace_key(session_id)
        payload = json.dumps(trace, ensure_ascii=False, default=str)
        pipe = self._redis.pipeline(transaction=True)
        pipe.zadd(key, {payload: timestamp})
        pipe.expire(key, self.ttl_seconds)
        pipe.zadd(f"{self._prefix}:sessions", {session_id: timestamp})
        pipe.expire(f"{self._prefix}:sessions", self.ttl_seconds)
        pipe.execute()

    def list_traces(self, session_id: str | None = None) -> list[dict]:
        if session_id is None:
            records: list[dict] = []
            for sid in self.list_trace_sessions():
                records.extend(self.list_traces(sid))
            records.sort(key=lambda item: (float(item.get("timestamp", 0)), item.get("call_id", "")))
            return records
        raw = self._redis.zrange(self._trace_key(session_id), 0, -1)
        return [json.loads(item) for item in raw]

    def list_trace_sessions(self) -> list[str]:
        key = f"{self._prefix}:sessions"
        self._redis.zremrangebyscore(key, "-inf", time.time() - self.ttl_seconds)
        return list(self._redis.zrange(key, 0, -1))

    def reset_traces(self) -> None:
        sessions = self.list_trace_sessions()
        keys = [self._trace_key(session_id) for session_id in sessions]
        keys.append(f"{self._prefix}:sessions")
        self._redis.delete(*keys)

    def rate_limit_allow(
        self, key: str, limit: int, window_seconds: float = 60.0
    ) -> bool:
        redis_key = f"{self._prefix}:rate:{self._digest(key)}"
        return bool(
            self._redis.eval(
                self._RATE_LIMIT_SCRIPT,
                1,
                redis_key,
                max(1, int(window_seconds * 1000)),
                max(0, int(limit)),
                uuid.uuid4().hex,
            )
        )

    def claim_alert(
        self,
        session_id: str,
        rule: str,
        cooldown_seconds: int,
        alert: dict,
    ) -> bool:
        cooldown_key = f"{self._prefix}:alert-cooldown:{self._digest(f'{session_id}:{rule}')}"
        record = {**alert, "fired_at": time.time()}
        return bool(
            self._redis.eval(
                self._ALERT_SCRIPT,
                2,
                cooldown_key,
                f"{self._prefix}:alerts",
                max(1, int(cooldown_seconds * 1000)),
                json.dumps(record, ensure_ascii=False, default=str),
                self.max_alerts,
                self.ttl_seconds,
            )
        )

    def list_alerts(self, limit: int = 100) -> list[dict]:
        raw = self._redis.lrange(
            f"{self._prefix}:alerts", 0, max(0, min(int(limit), 1000) - 1)
        )
        return [json.loads(item) for item in raw]

    def reset_alerts(self) -> None:
        keys = list(self._redis.scan_iter(f"{self._prefix}:alert-cooldown:*"))
        keys.append(f"{self._prefix}:alerts")
        self._redis.delete(*keys)

    def close(self) -> None:
        self._redis.close()


def create_observability_store(backend_type: str | None = None) -> ObservabilityStore:
    backend = (backend_type or os.getenv("STATE_BACKEND", "local")).strip().lower()
    ttl = int(os.getenv("MYCODER_OBSERVABILITY_TTL_SECONDS", str(_DEFAULT_TTL_SECONDS)))
    max_alerts = int(
        os.getenv("MYCODER_OBSERVABILITY_MAX_ALERTS", str(_DEFAULT_MAX_ALERTS))
    )
    if backend == "redis":
        return RedisObservabilityStore(
            os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            ttl_seconds=ttl,
            max_alerts=max_alerts,
        )
    path = os.getenv("MYCODER_OBSERVABILITY_PATH")
    if not path:
        path = str(Path.cwd() / ".mycoder" / "api_state.db")
    return SQLiteObservabilityStore(path, ttl_seconds=ttl, max_alerts=max_alerts)
