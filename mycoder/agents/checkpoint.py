"""Checkpoint persistence for long multi-step orchestrations (断点续跑).

A long task can span dozens of subagent steps. If the process dies at step N,
re-running from the start wastes steps 1..N-1 and risks re-applying side
effects. CheckpointStore persists, per session:

  * the plan (the decomposed assignments) — so resume uses the SAME plan, not a
    re-decomposed one that may differ;
  * each step's result envelope keyed by assignment id.

On resume, steps that already succeeded are skipped and execution continues
from the first failed / never-run step. Results are stored as plain dicts
(envelope.model_dump()) and re-validated into envelopes on load.
"""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any

from ..contracts.envelope import SubagentResultEnvelope

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


class CheckpointStore:
    """Per-session JSON checkpoints under ~/.mycoder/checkpoints/."""

    def __init__(self, base_dir: str | Path | None = None) -> None:
        self.base_dir = Path(
            base_dir or (Path.home() / ".mycoder" / "checkpoints")
        ).resolve()
        self._lock = threading.RLock()

    def _path(self, session_id: str) -> Path:
        safe = _SAFE.sub("-", session_id or "unknown")[:100]
        return self.base_dir / f"{safe}.json"

    # ---------------------------------------------------------------- plan

    def save_plan(self, session_id: str, task: str, assignments: list[dict]) -> None:
        """Persist the plan (keeps any results already recorded).

        The runtime-injected ``executor`` field is dropped — it's a live
        callable that can't be serialized and shouldn't be restored; on resume,
        unfinished steps run through the normal subagent path.
        """
        data = self._load_raw(session_id)
        data["task"] = task
        data["assignments"] = [
            {k: v for k, v in a.items() if k != "executor"} for a in assignments
        ]
        self._write(session_id, data)

    # -------------------------------------------------------------- results

    def record_result(
        self, session_id: str, assign_id: str, envelope: dict[str, Any]
    ) -> None:
        """Persist one step's result envelope (plain dict)."""
        data = self._load_raw(session_id)
        data["results"][assign_id] = envelope
        self._write(session_id, data)

    # ----------------------------------------------------------------- load

    def load(self, session_id: str) -> dict | None:
        """Return {task, assignments, results:{assign_id: envelope}} or None.

        Results are re-validated into SubagentResultEnvelope objects so the
        orchestrator can skip completed steps directly.
        """
        data = self._load_raw(session_id)
        if not data.get("assignments"):
            return None
        results: dict[str, SubagentResultEnvelope] = {}
        for aid, raw in (data.get("results") or {}).items():
            try:
                results[aid] = SubagentResultEnvelope.model_validate(raw)
            except Exception:  # noqa: BLE001 - a corrupt entry is dropped
                continue
        return {"task": data.get("task"), "assignments": data["assignments"], "results": results}

    def clear(self, session_id: str) -> None:
        self._path(session_id).unlink(missing_ok=True)

    def summary(self, session_id: str) -> dict | None:
        """Return API-safe progress metadata without exposing result payloads."""
        checkpoint = self.load(session_id)
        if checkpoint is None:
            return None
        assignment_ids = []
        for index, item in enumerate(checkpoint["assignments"]):
            name = str(item.get("subagent_name") or "task")
            assignment_ids.append(str(item.get("id") or f"{name}-{index + 1}"))
        completed = sorted(checkpoint["results"])
        completed_set = set(completed)
        remaining = [aid for aid in assignment_ids if aid not in completed_set]
        return {
            "task": checkpoint.get("task") or "",
            "assignment_count": len(assignment_ids),
            "completed_count": len(completed),
            "completed_steps": completed,
            "remaining_steps": remaining,
        }

    # -------------------------------------------------------------- helpers

    def _load_raw(self, session_id: str) -> dict:
        with self._lock:
            path = self._path(session_id)
            if not path.exists():
                return {"task": "", "assignments": [], "results": {}}
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {"task": "", "assignments": [], "results": {}}

    def _write(self, session_id: str, data: dict) -> None:
        with self._lock:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            target = self._path(session_id)
            temporary = target.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
            temporary.write_text(
                json.dumps(data, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            os.replace(temporary, target)


class RedisCheckpointStore(CheckpointStore):
    """Redis checkpoint store shared by every API worker.

    Plan and per-step results use separate keys, so recording one completed
    step is a single atomic HSET instead of a read/modify/write JSON race.
    """

    def __init__(self, url: str, ttl: int = 24 * 60 * 60) -> None:
        import redis

        self._redis = redis.from_url(url, decode_responses=True)
        self.ttl = ttl

    @staticmethod
    def _safe(session_id: str) -> str:
        return _SAFE.sub("-", session_id or "unknown")[:100]

    def _plan_key(self, session_id: str) -> str:
        return f"mycoder:checkpoint:{self._safe(session_id)}:plan"

    def _results_key(self, session_id: str) -> str:
        return f"mycoder:checkpoint:{self._safe(session_id)}:results"

    def save_plan(self, session_id: str, task: str, assignments: list[dict]) -> None:
        payload = {
            "task": task,
            "assignments": [
                {k: v for k, v in item.items() if k != "executor"}
                for item in assignments
            ],
        }
        with self._redis.pipeline() as pipe:
            pipe.set(self._plan_key(session_id), json.dumps(payload, ensure_ascii=False), ex=self.ttl)
            pipe.expire(self._results_key(session_id), self.ttl)
            pipe.execute()

    def record_result(
        self, session_id: str, assign_id: str, envelope: dict[str, Any]
    ) -> None:
        key = self._results_key(session_id)
        with self._redis.pipeline() as pipe:
            pipe.hset(key, assign_id, json.dumps(envelope, ensure_ascii=False, default=str))
            pipe.expire(key, self.ttl)
            pipe.execute()

    def load(self, session_id: str) -> dict | None:
        plan_raw = self._redis.get(self._plan_key(session_id))
        if not plan_raw:
            return None
        try:
            plan = json.loads(plan_raw)
        except (json.JSONDecodeError, TypeError):
            return None
        if not plan.get("assignments"):
            return None
        results: dict[str, SubagentResultEnvelope] = {}
        for aid, raw in self._redis.hgetall(self._results_key(session_id)).items():
            try:
                results[aid] = SubagentResultEnvelope.model_validate_json(raw)
            except Exception:  # noqa: BLE001 - corrupt entries are isolated
                continue
        return {"task": plan.get("task"), "assignments": plan["assignments"], "results": results}

    def clear(self, session_id: str) -> None:
        self._redis.delete(self._plan_key(session_id), self._results_key(session_id))
