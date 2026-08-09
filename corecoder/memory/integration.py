"""Memory integration wiring for the core agent loop.

Three settlement / injection points:

  * planning_guard  — before todo_write, a relevant-memory section is injected
                      so the fresh plan is informed by history;
  * Self-Correction — when run_with_correction recovers from a failure
                      (attempt > 0 and then success) a PatternMemory is saved;
  * plan completion — when the last todo step is marked done, the finished
                      plan is distilled into a `decision` memory.

All settlement is best-effort: a failure here must never break the agent's
tool execution, so every hook is wrapped in try/except.
"""

from __future__ import annotations

import logging
from typing import Any

from .prompt import MAX_MEMORY_TOKENS, MemoryPromptInjector
from .retriever import HybridRetriever
from .store import MemoryStore, get_store
from .types import PatternMemory

log = logging.getLogger(__name__)


def settle_pattern_memory(
    fn_name: str,
    strategy,
    params: dict | None,
    kwargs: dict | None,
    store: MemoryStore | None = None,
) -> str | None:
    """Persist a PatternMemory for a recovered failure. Returns the memory id."""
    store = store or get_store()
    strategy_name = getattr(strategy, "value", str(strategy))
    trigger = f"{fn_name} 执行失败"
    action = f"采用 {strategy_name} 策略重试"
    outcome = f"重试参数: {params or {}}"
    pattern = PatternMemory(
        content=f"{trigger}，{action}，{outcome}。",
        trigger=trigger,
        action=action,
        outcome=outcome,
        success_score=1.0,
        scope="project",
        source="auto",
        confidence=0.8,
    )
    return store.save(pattern)


def settle_plan_decision(plan, store: MemoryStore | None = None) -> str | None:
    """Distill a finished TaskPlan into a `decision` memory."""
    store = store or get_store()
    done = [t.id for t in plan.items if t.status == "done"]
    content = f"计划完成：{plan.goal}（步骤: {', '.join(done) or '—'}）"
    from .types import MemoryEntry

    entry = MemoryEntry(
        content=content,
        type="decision",
        scope="project",
        source="auto",
        confidence=0.7,
        metadata={"plan_id": plan.created_at},
    )
    return store.save(entry)


class MemoryIntegration:
    """Wires the memory system into planner/correction/todo hooks.

    Construct once (CLI startup) with a store; the same instance is handed to
    Agent so the memory tools share one database. install() is idempotent.
    """

    def __init__(
        self,
        store: MemoryStore | None = None,
        retriever: HybridRetriever | None = None,
        max_tokens: int = MAX_MEMORY_TOKENS,
    ):
        self.store = store or get_store()
        self.retriever = retriever or HybridRetriever(self.store)
        self.max_tokens = max_tokens
        self._installed = False

    def install(self) -> "MemoryIntegration":
        if self._installed:
            return self
        from ..planner import set_memory_injector, set_plan_complete_hook
        from ..tools.correction import set_recovery_hook

        set_memory_injector(self._inject_for_plan)
        set_plan_complete_hook(self._on_plan_complete)
        set_recovery_hook(self._on_recovery)
        self._installed = True
        log.debug("memory integration installed (backend=%s)", self.store.vector_backend_name)
        return self

    # -------------------------------------------------------------- hooks
    def _inject_for_plan(self, query: str) -> str:
        try:
            return MemoryPromptInjector(
                self.retriever, max_tokens=self.max_tokens
            ).build_memory_section(query)
        except Exception as exc:  # noqa: BLE001 - never block planning
            log.warning("memory injection failed: %s", exc)
            return ""

    def _on_plan_complete(self, plan) -> None:
        try:
            settle_plan_decision(plan, self.store)
        except Exception as exc:  # noqa: BLE001
            log.warning("plan decision settlement failed: %s", exc)

    def _on_recovery(self, **context: Any) -> None:
        try:
            settle_pattern_memory(
                context.get("fn_name", "tool"),
                context.get("strategy"),
                context.get("params"),
                context.get("kwargs"),
                self.store,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("pattern settlement failed: %s", exc)
