"""Dependency injection for the service layer.

  * get_state_backend()  -> process-wide singleton StateBackend.
  * get_orchestrator()   -> returns a builder; each call constructs a FRESH
                            Orchestrator bound to one session, with the state
                            backend injected.
  * get_default_llm()    -> lazily-built LLM from env config (None if no key).
  * get_checkpoint_store() -> process-wide file-backed orchestration checkpoints.

How the state backend reaches the Orchestrator WITHOUT touching
orchestrator.py (zero-intrusion): it is injected through a PersistentBlackboard
— the orchestrator's own state container and its first constructor argument.
The blackboard loads its initial snapshot from the backend and writes every
mutation back, so Orchestrator's constructor needs no new parameter and every
existing caller keeps working unchanged.
"""

from __future__ import annotations

import os
from collections.abc import Callable

from fastapi import Depends

from mycoder.agent_factory import AgentFactory
from mycoder.agents.blackboard import Blackboard
from mycoder.agents.checkpoint import CheckpointStore, RedisCheckpointStore
from mycoder.agents.orchestrator import Orchestrator
from mycoder.agents.planner import TaskPlanner
from mycoder.config import Config
from mycoder.observability.alerts import AlertManager
from mycoder.observability.budget import TokenBudgetGuard
from mycoder.observability.ratelimit import RateLimiter
from mycoder.observability.store import ObservabilityStore, create_observability_store
from mycoder.observability.trace import LLMTracer
from mycoder.observability.tool_trace import ToolTracer
from mycoder.tools import ALL_TOOLS
from mycoder.tools import build_scoped_tools

from .state_backend import StateBackend, create_state_backend

_state_backend: StateBackend | None = None
_llm = None
_tracer: LLMTracer | None = None
_tool_tracer: ToolTracer | None = None
_checkpoint_store: CheckpointStore | None = None
_observability_store: ObservabilityStore | None = None
_alert_manager: AlertManager | None = None
_rate_limiter: RateLimiter | None = None
_rate_limiter_ready = False


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


def get_observability_store() -> ObservabilityStore:
    """Shared SQLite/Redis state for traces, rate windows and alerts."""
    global _observability_store
    if _observability_store is None:
        _observability_store = create_observability_store()
    return _observability_store


def get_alert_manager() -> AlertManager:
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager(store=get_observability_store())
    return _alert_manager


def get_tracer() -> LLMTracer:
    """Recorder backed by process-independent SQLite/Redis trace state."""
    global _tracer
    if _tracer is None:
        _tracer = LLMTracer(store=get_observability_store())
        _tracer.attach_alert_manager(get_alert_manager())
    return _tracer


def get_tool_tracer() -> ToolTracer:
    """Durable action-layer trace shared by API workers and subagents."""
    global _tool_tracer
    if _tool_tracer is None:
        _tool_tracer = ToolTracer(store=get_observability_store())
    return _tool_tracer


def get_rate_limiter() -> RateLimiter | None:
    """Configured distributed limiter; None keeps the endpoint unlimited."""
    global _rate_limiter, _rate_limiter_ready
    if not _rate_limiter_ready:
        # Avoid opening SQLite/Redis when the feature is disabled.
        if os.getenv("MYCODER_RATE_LIMIT", "").strip():
            _rate_limiter = RateLimiter.from_env(store=get_observability_store())
        _rate_limiter_ready = True
    return _rate_limiter


def reset_observability_runtime() -> None:
    """Close/reset process handles without deleting persisted state (tests/reload)."""
    global _tracer, _tool_tracer, _observability_store, _alert_manager, _rate_limiter, _llm
    global _rate_limiter_ready
    _llm = None  # it owns the old tracer reference
    if _observability_store is not None:
        try:
            _observability_store.close()
        except Exception:  # noqa: BLE001 - teardown is best-effort
            pass
    _tracer = None
    _tool_tracer = None
    _alert_manager = None
    _rate_limiter = None
    _rate_limiter_ready = False
    _observability_store = None


def get_checkpoint_store() -> CheckpointStore:
    """Process-wide checkpoint store used by every API orchestrator."""
    global _checkpoint_store
    if _checkpoint_store is None:
        if os.getenv("STATE_BACKEND", "local").strip().lower() == "redis":
            _checkpoint_store = RedisCheckpointStore(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
        else:
            _checkpoint_store = CheckpointStore(base_dir=os.getenv("MYCODER_CHECKPOINT_DIR") or None)
    return _checkpoint_store


def get_default_llm():
    """Lazily-built LLM from env config, or None when no API key is set. The
    shared LLMTracer is attached so every LLM call in the API is traced."""
    global _llm
    if _llm is not None:
        return _llm
    cfg = Config.from_env()
    if not cfg.api_key:
        return None
    from mycoder.llm import LLM, LiteLLM

    llm_cls = LiteLLM if cfg.provider == "litellm" else LLM
    llm_options = {}
    if cfg.provider == "deepseek" and cfg.thinking != "auto":
        llm_options["extra_body"] = {"thinking": {"type": cfg.thinking}}
    _llm = llm_cls(
        model=cfg.model,
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        provider=cfg.provider,
        tool_dialect=cfg.tool_dialect,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        tracer=get_tracer(),
        caller="api",
        **llm_options,
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
        session_id: str,
        *,
        llm=None,
        tools=None,
        budget_guard: TokenBudgetGuard | None = None,
        checkpoint_store: CheckpointStore | None = None,
        workspace_root: str | None = None,
        reasoning_strategy: str | None = None,
        sandbox_policy: str = "interactive",
        sandbox_image: str | None = None,
        sandbox_user: str = "sandbox",
        soft_budget_ratio: float | None = None,
    ) -> Orchestrator:
        from mycoder.memory.experience import remember_replan
        from mycoder.model_router import build_model_factory

        blackboard = PersistentBlackboard(state_backend, session_id)
        llm = llm if llm is not None else get_default_llm()
        manager = None
        if tools is None and workspace_root is not None:
            tools, manager = build_scoped_tools(
                workspace_root,
                session_id,
                sandbox_policy=sandbox_policy,
                sandbox_image=sandbox_image,
                sandbox_user=sandbox_user,
            )
        runtime_tools = tools if tools is not None else ALL_TOOLS
        agent_factory = AgentFactory.from_defaults(
            llm=llm,
            tools=runtime_tools,
            max_context_tokens=Config.from_env().max_context_tokens,
            budget_guard=budget_guard,
            soft_budget_ratio=soft_budget_ratio,
            tool_tracer=get_tool_tracer(),
        )
        orchestrator = Orchestrator(
            blackboard=blackboard,
            llm=llm,
            tools=runtime_tools,
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
            experience_store=agent_factory.experience_store or remember_replan,
            # Checkpointed plans/results survive API worker or process failure;
            # POST /v1/agent/run with resume=true reuses them.
            checkpoint_store=checkpoint_store or get_checkpoint_store(),
            reasoning_strategy=reasoning_strategy,
            tool_tracer=get_tool_tracer(),
        )
        orchestrator.agent_factory = agent_factory
        # The API worker owns and tears down this per-run manager.
        orchestrator._sandbox_manager = manager  # noqa: SLF001
        return orchestrator

    return _build
