"""Re-planning experience settlement (对标 Hermes 经验沉淀).

Every (subagent, deviation, strategy, recovered) record produced by the
orchestrator's dynamic re-planning is persisted to the memory DB, so a future
session can retrieve "this deviation → this strategy recovered" through agentic
RAG (memory_search). Shared by the CLI and the API service layer.
"""

from __future__ import annotations

from typing import Any


def remember_replan(record: dict[str, Any], store=None) -> str | None:
    """Persist one re-planning experience; returns the memory id or None.

    Best-effort: a missing memory backend (or any failure) yields None and
    never breaks the orchestrator run. ``store`` defaults to the process-wide
    singleton (the CLI's already-configured store is reused when present).
    """
    try:
        from .store import get_store
        from .types import MemoryEntry

        store = store or get_store()
        status = "已恢复" if record.get("recovered") else "未恢复"
        return store.save(
            MemoryEntry(
                content=(
                    f"重规划经验：{record.get('subagent', '?')} "
                    f"{record.get('deviation', '?')} → 策略 "
                    f"{record.get('strategy', '?')}（{status}）"
                ),
                type="pattern",
                scope="project",
                source="auto",
                confidence=0.6,
                metadata=record,
            )
        )
    except Exception:  # noqa: BLE001 - experience write is best-effort
        return None
