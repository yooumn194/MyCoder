"""Dependency injection for the service layer.

  * get_state_backend()  -> process-wide singleton StateBackend.
  * get_orchestrator()   -> returns a builder; each call constructs a FRESH
                            Orchestrator bound to one session, with the state
                            backend injected.
  * get_default_llm()    -> lazily-built LLM from env config (None if no key).

How the state backend reaches the Orchestrator WITHOUT touching
orchestrator.py (zero-intrusion): it is injected through a PersistentBlackboard
— the orchestrator's own state container and its first constructor argument.
The blackboard loads its initial snapshot from the backend and writes every
mutation back, so Orchestrator's constructor needs no new parameter and every
existing caller keeps working unchanged.
"""

from __future__ import annotations

from collections.abc import Callable

from fastapi import Depends

from mycoder.agents.blackboard import Blackboard
from mycoder.agents.orchestrator import Orchestrator
from mycoder.agents.planner import TaskPlanner
from mycoder.config import Config
from mycoder.observability.budget import TokenBudgetGuard
from mycoder.observability.trace import LLMTracer
from mycoder.tools import ALL_TOOLS

from .state_backend import StateBackend, create_state_backend

_state_backend: StateBackend | None = None
_llm = None
# Process-wide LLM trace store: every session's calls land here, and /cost reads
# it. Resets on process restart (in-memory by design).
_tracer: LLMTracer | None = None


class PersistentBlackboard(Blackboard):
    """A Blackboard that survives the HTTP boundary by delegating to a
    StateBackend. Blackboard's TTL / subscriber logic is reused unchanged; only
    the put/get/query hooks gain a load-before / persist-after step."""

    def __init__(
        self,
        state_backend: StateBackend | None,
        session_id: str,
        ttl_seconds: int = 300,
    ) -> None:
        super().__init__(ttl_seconds)
        self._backend = state_backend
        self._session_id = session_id
        self._loaded = False

    async def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if self._backend is not None:
            snapshot = await self._backend.get_blackboard(self._session_id)
            if snapshot:
                self._store = {k: v for k, v in snapshot.items() if isinstance(v, dict)}
        self._loaded = True

    async def _persist(self) -> None:
        if self._backend is not None:
            await self._backend.save_blackboard(self._session_id, self._store)

    async def put(self, task_id: str, key: str, value, ttl: int | None = None) -> None:
        await self._ensure_loaded()
        await super().put(task_id, key, value, ttl)
        await self._persist()

    async def get(self, task_id: str, key: str):
        await self._ensure_loaded()
        return await super().get(task_id, key)

    async def query(self, task_id: str, prefix: str) -> dict:
        await self._ensure_loaded()
        return await super().query(task_id, prefix)


def get_state_backend() -> StateBackend:
    """Process-wide singleton backend (created once on first use)."""
    global _state_backend
    if _state_backend is None:
        _state_backend = create_state_backend()
    return _state_backend


def get_tracer() -> LLMTracer:
    """Process-wide LLM trace store shared by every session."""
    global _tracer
    if _tracer is None:
        _tracer = LLMTracer()
    return _tracer


def get_default_llm():
    """Lazily-built LLM from env config, or None when no API key is set. The
    shared LLMTracer is attached so every LLM call in the API is traced."""
    global _llm
    if _llm is not None:
        return _llm
    cfg = Config.from_env()
    if not cfg.api_key:
        return None
    from mycoder.llm import LLM

    _llm = LLM(
        model=cfg.model,
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        tracer=get_tracer(),
        caller="api",
    )
    return _llm


def get_orchestrator(
    state_backend: StateBackend = Depends(get_state_backend),
) -> Callable[..., Orchestrator]:
    """Return a builder; calling it constructs a fresh Orchestrator bound to a
    session (a new PersistentBlackboard, the default LLM and the full tool set).

    A new instance per call keeps per-session state isolated — nothing is shared
    between concurrent requests except the (thread-safe) state backend."""

    def _build(
        session_id: str, *, llm=None, tools=None, budget_guard: TokenBudgetGuard | None = None
    ) -> Orchestrator:
        from mycoder.memory.experience import remember_replan
        from mycoder.model_router import build_model_factory

        blackboard = PersistentBlackboard(state_backend, session_id)
        llm = llm if llm is not None else get_default_llm()
        return Orchestrator(
            blackboard=blackboard,
            llm=llm,
            tools=tools if tools is not None else ALL_TOOLS,
            # LLM-driven task decomposition: with no key the planner degrades
            # to the single-explorer fallback, never raising.
            planner=TaskPlanner(llm=llm),
            # Token-budget enforcement; None = no budget (backward compatible).
            budget_guard=budget_guard,
            # P2 model-tier routing (cost): sub-agents get a tier-appropriate
            # model per config/model_routing.yaml instead of the shared LLM.
            model_factory=build_model_factory(llm),
            # P1 re-planning experience: deviation playbooks persist to the
            # memory DB (best-effort; no memory backend -> no-op) so API
            # sub-agent recovery lessons are reusable across sessions.
            experience_store=remember_replan,
        )

    return _build
