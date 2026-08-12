"""MemoryStore: dual-SQLite persistence for the Phase 5 memory system.

Two databases with identical schema:

  * project -> {project_root}/.corecoder/memory.db   (scope='project')
  * global  -> ~/.corecoder/memory.db                (scope='global')

Each db holds:
  * memories     — main table (id is a uuid; the implicit rowid links to FTS)
  * memories_fts — FTS5 (tokenize='ascii'), *manually* managed at the app
                   layer: content is jieba/fallback-tokenized before insert, so
                   the FTS5 tokenizer only sees a space-joined token stream.
  * embeddings   — vector index. Two interchangeable backends:
                     - sqlite-vec `vec0` (fast; needs the loadable extension)
                     - numpy brute-force over a BLOB column (zero extra deps)

The numpy backend keeps the hybrid path (and RRF) fully functional when
sqlite-vec is absent — that is the required graceful degradation. All heavy
resources (connections, embedder model) are created lazily.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Iterable

from .embedder import Embedder, get_embedder
from .security import filter_sensitive
from .tokenizer import build_match_query, tokenize_chinese
from .types import (
    DEDUP_COSINE,
    EMBED_DIM,
    MemoryEntry,
    new_id,
)

log = logging.getLogger(__name__)

try:  # numpy is the only hard assumption; without it vectors degrade to none
    import numpy as np
except ImportError:  # pragma: no cover - environment without numpy
    np = None  # type: ignore[assignment]

_BASE_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    type TEXT NOT NULL CHECK(type IN ('user','feedback','project','pattern','reference','decision','fact')),
    scope TEXT NOT NULL CHECK(scope IN ('project','global')),
    source TEXT NOT NULL CHECK(source IN ('auto','user','confirmed')),
    confidence REAL NOT NULL DEFAULT 0.5,
    access_count INTEGER NOT NULL DEFAULT 0,
    last_accessed REAL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    deprecated_by TEXT,
    metadata TEXT
);
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(content, type, scope, tokenize='ascii');
CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(scope);
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type);
"""

_MEM_COLS = (
    "id", "content", "type", "scope", "source", "confidence",
    "access_count", "last_accessed", "created_at", "updated_at",
    "deprecated_by", "metadata",
)


def _json_dumps(metadata: dict) -> str:
    return json.dumps(metadata or {}, ensure_ascii=False, default=str)


def _json_loads(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


# ---------------------------------------------------------------------------
# Vector backends
# ---------------------------------------------------------------------------
class _VectorBackend:
    name = "base"

    def create_tables(self, conn: sqlite3.Connection) -> None: ...
    def insert(self, conn: sqlite3.Connection, mem_id: str, vector) -> None: ...
    def delete(self, conn: sqlite3.Connection, mem_id: str) -> None: ...
    def search(self, conn: sqlite3.Connection, query_vec, limit: int): ...


class _NumpyVectorBackend(_VectorBackend):
    """Brute-force cosine search over a BLOB column — the zero-dep fallback."""

    name = "numpy"

    def create_tables(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings ("
            "id TEXT PRIMARY KEY, vector BLOB NOT NULL)"
        )

    def insert(self, conn: sqlite3.Connection, mem_id: str, vector) -> None:
        blob = np.asarray(vector, dtype=np.float32).tobytes()
        conn.execute(
            "INSERT OR REPLACE INTO embeddings(id, vector) VALUES (?, ?)",
            (mem_id, blob),
        )

    def delete(self, conn: sqlite3.Connection, mem_id: str) -> None:
        conn.execute("DELETE FROM embeddings WHERE id=?", (mem_id,))

    def search(self, conn: sqlite3.Connection, query_vec, limit: int):
        rows = conn.execute("SELECT id, vector FROM embeddings").fetchall()
        if not rows:
            return []
        qv = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        ids = [r["id"] for r in rows]
        mat = np.vstack(
            [np.frombuffer(r["vector"], dtype=np.float32) for r in rows]
        )
        qn = np.linalg.norm(qv, axis=1) + 1e-9
        mn = np.linalg.norm(mat, axis=1) + 1e-9
        sim = (mat @ qv.T).flatten() / (mn * qn)
        dist = (1.0 - sim).astype(np.float64)
        order = np.argsort(dist)[:limit]
        return [(ids[i], float(dist[i])) for i in order]


class _Vec0VectorBackend(_VectorBackend):
    """sqlite-vec `vec0` KNN — used only when the extension is loadable.

    Falls back to the exact `vec_distance_cosine` scan if the KNN `MATCH`
    syntax is unavailable on the running build.
    """

    name = "vec0"

    def __init__(self) -> None:
        self._available: bool | None = None

    def _ensure(self, conn: sqlite3.Connection) -> None:
        if self._available is None:
            try:
                import sqlite_vec  # type: ignore

                conn.enable_load_extension(True)
                sqlite_vec.load(conn)
                self._available = True
            except Exception:
                self._available = False
        if not self._available:
            raise RuntimeError("sqlite-vec extension unavailable")

    def create_tables(self, conn: sqlite3.Connection) -> None:
        self._ensure(conn)
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS embeddings USING vec0("
            f"id TEXT PRIMARY KEY, vector FLOAT[{EMBED_DIM}])"
        )

    def insert(self, conn: sqlite3.Connection, mem_id: str, vector) -> None:
        self._ensure(conn)
        encoded = json.dumps(np.asarray(vector, dtype=np.float32).tolist())
        conn.execute(
            "INSERT OR REPLACE INTO embeddings(id, vector) VALUES (?, ?)",
            (mem_id, encoded),
        )

    def delete(self, conn: sqlite3.Connection, mem_id: str) -> None:
        self._ensure(conn)
        conn.execute("DELETE FROM embeddings WHERE id=?", (mem_id,))

    def search(self, conn: sqlite3.Connection, query_vec, limit: int):
        self._ensure(conn)
        encoded = json.dumps(np.asarray(query_vec, dtype=np.float32).tolist())
        try:
            rows = conn.execute(
                "SELECT id, vec_distance_cosine(vector, ?) AS d FROM embeddings "
                "WHERE vector MATCH ? ORDER BY d LIMIT ?",
                (encoded, encoded, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            # brute-force exact scan (KNN MATCH syntax unavailable)
            rows = conn.execute(
                "SELECT id, vec_distance_cosine(vector, ?) AS d FROM embeddings "
                "ORDER BY d LIMIT ?",
                (encoded, limit),
            ).fetchall()
        return [(r["id"], float(r["d"])) for r in rows]


def _make_vector_backend() -> _VectorBackend:
    """vec0 when sqlite-vec loads, else the numpy brute-force backend."""
    if np is None:
        return _NumpyVectorBackend()  # numpy missing -> vectors just won't insert
    try:
        import sqlite_vec  # noqa: F401

        probe = sqlite3.connect(":memory:")
        probe.row_factory = sqlite3.Row
        backend = _Vec0VectorBackend()
        backend._ensure(probe)
        backend.create_tables(probe)
        probe.close()
        return backend
    except Exception:
        return _NumpyVectorBackend()


# ---------------------------------------------------------------------------
# Per-database connection wrapper
# ---------------------------------------------------------------------------
class _DB:
    def __init__(self, scope: str, path: Path, vector: _VectorBackend):
        self.scope = scope
        self.path = path
        self._vector = vector
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.path))
            self._conn.row_factory = sqlite3.Row
            self._init_schema()
        return self._conn

    def _init_schema(self) -> None:
        conn = self._conn
        assert conn is not None
        conn.executescript(_BASE_SCHEMA)
        self._vector.create_tables(conn)
        conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


# ---------------------------------------------------------------------------
# MemoryStore
# ---------------------------------------------------------------------------
class MemoryStore:
    def __init__(
        self,
        project_dir: str | Path | None = None,
        global_dir: str | Path | None = None,
        embedder: Embedder | None = None,
        filter_sensitive: bool = True,
        vector_backend: _VectorBackend | None = None,
    ):
        """Both db roots are resolved lazily; passing tmp_path in tests keeps
        the real ~/.corecoder untouched."""
        project_root = Path(project_dir or os.getcwd()).expanduser().resolve()
        global_root = Path(
            global_dir or (Path.home() / ".corecoder")
        ).expanduser().resolve()
        self.embedder = embedder
        self.filter_sensitive = filter_sensitive
        self.project_db_path = project_root / ".corecoder" / "memory.db"
        self.global_db_path = global_root / "memory.db"
        backend = vector_backend if vector_backend is not None else _make_vector_backend()
        self._vector = backend
        self._project_db = _DB("project", self.project_db_path, backend)
        self._global_db = _DB("global", self.global_db_path, backend)

    # ------------------------------------------------------------- lifecycle
    def _db_for(self, scope: str) -> _DB:
        return self._project_db if scope == "project" else self._global_db

    def _dbs(self, scope: str | None = None) -> list[_DB]:
        return [self._db_for(scope)] if scope else [self._project_db, self._global_db]

    def close(self) -> None:
        self._project_db.close()
        self._global_db.close()

    @property
    def vector_backend_name(self) -> str:
        return self._vector.name

    # ---------------------------------------------------------------- writes
    def save(self, entry: MemoryEntry, *, source: str | None = None, dedup: bool = True) -> str:
        """Persist a memory. By default near-duplicate content (>DEDUP_COSINE)
        is merged into the existing row instead of creating a new one.

        `dedup=False` forces a fresh row under the entry's id — used by the
        document layer (document.py) where two chunks may legitimately share
        wording and content-similarity dedup would break the deterministic
        `doc:{doc_id}:{index}` id scheme.
        """
        if self.filter_sensitive:
            entry.content = filter_sensitive(entry.content)
        if source is not None:
            entry.source = source
        if not entry.id:
            entry.id = new_id()
        if dedup:
            dup_id = self._find_duplicate(entry.content, exclude_id=entry.id)
            if dup_id:
                self.update(
                    dup_id,
                    content=entry.content,
                    type=entry.type,
                    confidence=entry.confidence,
                    metadata=entry.metadata,
                )
                return dup_id

        db = self._db_for(entry.scope)
        conn = db.conn
        tokenized = tokenize_chinese(entry.content)
        with conn:
            cur = conn.execute(
                f"INSERT INTO memories ({', '.join(_MEM_COLS)}) "
                f"VALUES ({', '.join('?' * len(_MEM_COLS))})",
                (
                    entry.id, entry.content, entry.type, entry.scope,
                    entry.source, entry.confidence, entry.access_count,
                    entry.last_accessed, entry.created_at, entry.updated_at,
                    entry.deprecated_by, _json_dumps(entry.metadata),
                ),
            )
            conn.execute(
                "INSERT INTO memories_fts(rowid, content, type, scope) "
                "VALUES (?, ?, ?, ?)",
                (cur.lastrowid, tokenized, entry.type, entry.scope),
            )
        self._save_vector(entry.id, entry.content, entry.scope)
        return entry.id

    def update(
        self,
        mem_id: str,
        *,
        content: str | None = None,
        type: str | None = None,
        source: str | None = None,
        confidence: float | None = None,
        metadata: dict | None = None,
        deprecated_by: str | None = None,
    ) -> str | None:
        entry = self.get(mem_id, touch=False)
        if entry is None:
            return None
        if content is not None:
            entry.content = filter_sensitive(content) if self.filter_sensitive else content
        if type is not None:
            entry.type = type
        if source is not None:
            entry.source = source
        if confidence is not None:
            entry.confidence = confidence
        if metadata is not None:
            entry.metadata = metadata
        if deprecated_by is not None:
            entry.deprecated_by = deprecated_by
        entry.updated_at = time.time()
        db = self._db_for(entry.scope)
        conn = db.conn
        tokenized = tokenize_chinese(entry.content)
        with conn:
            conn.execute(
                "UPDATE memories SET content=?, type=?, source=?, confidence=?, "
                "updated_at=?, deprecated_by=?, metadata=? WHERE id=?",
                (
                    entry.content, entry.type, entry.source, entry.confidence,
                    entry.updated_at, entry.deprecated_by,
                    _json_dumps(entry.metadata), mem_id,
                ),
            )
            conn.execute(
                "UPDATE memories_fts SET content=?, type=?, scope=? "
                "WHERE rowid = (SELECT rowid FROM memories WHERE id=?)",
                (tokenized, entry.type, entry.scope, mem_id),
            )
        self._save_vector(mem_id, entry.content, entry.scope)
        return mem_id

    def delete(self, mem_id: str) -> bool:
        removed = False
        for db in self._dbs():
            conn = db.conn
            row = conn.execute(
                "SELECT rowid FROM memories WHERE id=?", (mem_id,)
            ).fetchone()
            if row:
                with conn:
                    conn.execute("DELETE FROM memories WHERE id=?", (mem_id,))
                    conn.execute("DELETE FROM memories_fts WHERE rowid=?", (row["rowid"],))
                try:
                    self._vector.delete(conn, mem_id)
                except Exception:  # pragma: no cover - backend-specific
                    pass
                removed = True
        return removed

    def confirm(self, mem_id: str) -> bool:
        """Promote a memory to confirmed: source='confirmed', confidence=1.0.
        Confirmed (and user) memories never decay (see maintenance.decay)."""
        for db in self._dbs():
            exists = db.conn.execute(
                "SELECT 1 FROM memories WHERE id=?", (mem_id,)
            ).fetchone()
            if exists:
                with db.conn:
                    db.conn.execute(
                        "UPDATE memories SET source='confirmed', confidence=1.0, "
                        "updated_at=? WHERE id=?",
                        (time.time(), mem_id),
                    )
                return True
        return False

    # ----------------------------------------------------------------- reads
    def get(self, mem_id: str, *, touch: bool = False) -> MemoryEntry | None:
        for db in self._dbs():
            row = db.conn.execute(
                "SELECT * FROM memories WHERE id=?", (mem_id,)
            ).fetchone()
            if row:
                data = dict(row)
                data["metadata"] = _json_loads(data.get("metadata"))
                if touch:
                    self._touch(mem_id, db)
                return MemoryEntry.from_metadata(**data)
        return None

    def list(
        self,
        scope: str | None = None,
        type: str | None = None,
        limit: int = 50,
        include_deprecated: bool = False,
    ) -> list[MemoryEntry]:
        entries: list[MemoryEntry] = []
        conds, params = [], []
        if type:
            conds.append("type=?")
            params.append(type)
        if not include_deprecated:
            conds.append("deprecated_by IS NULL")
        where = (" WHERE " + " AND ".join(conds)) if conds else ""
        for db in self._dbs(scope):
            for row in db.conn.execute(
                f"SELECT * FROM memories{where} ORDER BY created_at DESC "
                f"LIMIT {int(limit)}",
                params,
            ):
                data = dict(row)
                data["metadata"] = _json_loads(data.get("metadata"))
                entries.append(MemoryEntry.from_metadata(**data))
        entries.sort(key=lambda e: e.created_at, reverse=True)
        return entries

    def list_by_metadata(self, key: str, value: str) -> list[MemoryEntry]:
        """Return entries whose metadata dict has metadata[key] == value.

        Uses a coarse SQL LIKE pre-filter on the JSON `metadata` column and
        re-verifies the parsed dict, so the match is exact even when the stored
        JSON escapes characters. Used by the document layer to enumerate a
        document's chunks for incremental reindexing.
        """
        pattern = f'%"{key}": "{value}"%'
        entries: list[MemoryEntry] = []
        for db in self._dbs():
            for row in db.conn.execute(
                "SELECT * FROM memories WHERE metadata LIKE ?", (pattern,)
            ):
                data = dict(row)
                data["metadata"] = _json_loads(data.get("metadata"))
                if data["metadata"].get(key) == value:
                    entries.append(MemoryEntry.from_metadata(**data))
        return entries

    def fetch_rows(self, mem_ids: Iterable[str]) -> dict[str, dict]:
        ids = list(mem_ids)
        if not ids:
            return {}
        placeholders = ", ".join("?" * len(ids))
        result: dict[str, dict] = {}
        for db in self._dbs():
            for row in db.conn.execute(
                f"SELECT * FROM memories WHERE id IN ({placeholders})", ids
            ):
                data = dict(row)
                data["metadata"] = _json_loads(data.get("metadata"))
                result[row["id"]] = data
        return result

    def mark_accessed(self, mem_ids: Iterable[str]) -> None:
        ids = list(mem_ids)
        if not ids:
            return
        now = time.time()
        placeholders = ", ".join("?" * len(ids))
        for db in self._dbs():
            with db.conn:
                db.conn.execute(
                    f"UPDATE memories SET access_count = access_count + 1, "
                    f"last_accessed = ? WHERE id IN ({placeholders})",
                    [now, *ids],
                )

    def _touch(self, mem_id: str, db: _DB) -> None:
        try:
            with db.conn:
                db.conn.execute(
                    "UPDATE memories SET access_count = access_count + 1, "
                    "last_accessed = ? WHERE id=?",
                    (time.time(), mem_id),
                )
        except Exception:  # pragma: no cover - defensive
            pass

    # ------------------------------------------------------------ dedup
    def _find_duplicate(self, content: str, exclude_id: str | None = None) -> str | None:
        if self.embedder is not None:
            vec = self._embed(content)
            if vec is not None:
                best_id, best_dist = None, float("inf")
                for db in self._dbs():
                    try:
                        hits = self._vector.search(db.conn, vec, limit=1)
                    except Exception:
                        continue
                    for mem_id, dist in hits:
                        if exclude_id and mem_id == exclude_id:
                            continue
                        if dist < best_dist:
                            best_id, best_dist = mem_id, dist
                if best_id is not None and best_dist < (1.0 - DEDUP_COSINE):
                    return best_id
        # exact-content fallback when there is no embedder / no vectors
        for db in self._dbs():
            row = db.conn.execute(
                "SELECT id FROM memories WHERE content = ? AND id != ? LIMIT 1",
                (content, exclude_id or ""),
            ).fetchone()
            if row:
                return row["id"]
        return None

    # ------------------------------------------------------------- retrieval
    def bm25_search(
        self, query: str, limit: int = 30, scope: str | None = None
    ) -> list[tuple[str, float]]:
        """FTS5 BM25 over both dbs. Returns [(id, score)] with score in 0..1
        (1 = best) — FTS5's raw rank is negative, normalized here."""
        tokenized = tokenize_chinese(query)
        if not tokenized:
            return []
        match_expr = build_match_query(tokenized)
        raw: list[tuple[str, float]] = []
        for db in self._dbs(scope):
            try:
                rows = db.conn.execute(
                    "SELECT m.id, memories_fts.rank AS rank "
                    "FROM memories_fts JOIN memories m ON m.rowid = memories_fts.rowid "
                    "WHERE memories_fts MATCH ? ORDER BY memories_fts.rank",
                    (match_expr,),
                ).fetchall()
            except sqlite3.OperationalError as exc:
                log.warning("FTS5 query failed (degrading to empty): %s", exc)
                continue
            raw.extend((r["id"], float(r["rank"])) for r in rows)
        if not raw:
            return []
        lo = min(r for _, r in raw)
        hi = max(r for _, r in raw)
        span = (hi - lo) or 1.0
        scored = [(mem_id, (hi - r) / span) for mem_id, r in raw]
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return scored[:limit]

    def embed_query(self, text: str):
        return self._embed(text)

    def vector_search(
        self, query_vec, limit: int = 30, scope: str | None = None
    ) -> list[tuple[str, float]]:
        qv = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        best: dict[str, float] = {}
        for db in self._dbs(scope):
            try:
                hits = self._vector.search(db.conn, qv, limit)
            except Exception as exc:
                log.warning("vector search unavailable in %s db: %s", db.scope, exc)
                continue
            for mem_id, dist in hits:
                if mem_id not in best or dist < best[mem_id]:
                    best[mem_id] = dist
        ordered = sorted(best.items(), key=lambda kv: kv[1])[:limit]
        return list(ordered)

    # ---------------------------------------------------------------- internals
    def _embed(self, text: str):
        if self.embedder is None or np is None:
            return None
        try:
            return np.asarray(self.embedder.embed(text), dtype=np.float32)
        except Exception as exc:
            log.warning("embedding failed, degrading to BM25-only: %s", exc)
            return None

    def _save_vector(self, mem_id: str, content: str, scope: str) -> None:
        vec = self._embed(content)
        if vec is None:
            return
        db = self._db_for(scope)
        try:
            self._vector.insert(db.conn, mem_id, vec)
        except Exception as exc:
            log.warning("vector insert failed for %s: %s", mem_id, exc)


# ---------------------------------------------------------------------------
# Convenience singletons (used by the CLI and the memory tools by default)
# ---------------------------------------------------------------------------
_store: MemoryStore | None = None


def get_store(config: dict | None = None) -> MemoryStore:
    global _store
    if _store is None:
        _store = MemoryStore(embedder=get_embedder(config))
    return _store


def reset_store() -> None:
    """Drop the process-wide singleton (used by tests to avoid leakage)."""
    global _store
    _store = None
