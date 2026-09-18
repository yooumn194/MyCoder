"""FastAPI service layer for MyCoder.

HTTP service endpoints + one background worker:

  * POST /v1/agent/run             — schedule an Orchestrator run and return
                                     immediately (BackgroundTasks, so the
                                     request never blocks on the model).
  * GET  /v1/agent/status/{id}     — read the session record the worker wrote.
                                     Includes resumable checkpoint progress.
  * GET  /health                   — liveness + backend connectivity + version.

The worker maps the orchestrator's RFC v1.0.1 envelope error codes
(CIRCUIT_BREAKER_OPEN / SUBAGENT_TIMEOUT) onto the API's structured error
types. A background task's exception never reaches the request, so the worker
also persists the translated error into the session record for GET /status;
the global exception handlers return the same shape for synchronous setup
failures (defense in depth).

Run:  uvicorn api.server:app --reload        (STATE_BACKEND unset -> SQLite)
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import uuid
from collections import Counter
from contextlib import suppress
from contextlib import asynccontextmanager
from typing import Literal
from pathlib import Path
from types import SimpleNamespace

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from structlog.contextvars import bind_contextvars

from mycoder import __version__
from mycoder.agents.checkpoint import CheckpointStore
from mycoder.contracts import RunContract
from mycoder.agents.orchestrator import OrchestrationStrategy
from mycoder.observability.budget import TokenBudgetExceeded, TokenBudgetGuard
from mycoder.sandbox import (
    BENCHMARK_VERIFY_CMD_ENV as _BENCHMARK_VERIFY_CMD_ENV,
    benchmark_verify_command,
    benchmark_verify_timeout,
)
from mycoder.sandbox.logger import get_logger
from mycoder.llm import ToolChoiceCapabilityError

from .dependencies import (
    get_alert_manager,
    get_checkpoint_store,
    get_default_llm,
    get_orchestrator,
    get_rate_limiter,
    get_state_backend,
    get_tool_tracer,
    get_tracer,
)
from .state_backend import RedisStateBackend, StateBackend
from .auth import Principal, audit, get_principal, scope_session_id
from mycoder.observability.ratelimit import RateLimiter

logger = get_logger("mycoder.api")

# Explicit override hook retained for tests/custom embedding. Normal API
# operation resolves a SQLite/Redis-backed limiter from dependencies.
RATE_LIMITER: RateLimiter | None = None


# --------------------------------------------------------------------------
# Structured error types raised by the service layer and handled globally.
# (They are API-level exceptions; the underlying modules signal the same
# conditions through envelope error codes, which the worker translates.)
# TokenBudgetExceeded is the shared class from mycoder.observability.budget
# (raised by TokenBudgetGuard); CircuitBreakerOpen / SandboxTimeout are API-level.
# --------------------------------------------------------------------------
class CircuitBreakerOpen(Exception):
    """A subagent was skipped because its circuit breaker is open."""


class SandboxTimeout(Exception):
    """A sandbox / subagent execution exceeded its hard timeout."""


class PatchRequired(Exception):
    """A benchmark run reached a terminal state without repository changes."""


_CIRCUIT_BREAKER_CODES = {"CIRCUIT_BREAKER_OPEN"}
_TIMEOUT_CODES = {"SUBAGENT_TIMEOUT"}
_BUDGET_CODES = {"TOKEN_BUDGET_EXCEEDED"}
_PATCH_CODES = {"PATCH_REQUIRED", "PATCH_SCOPE_VIOLATION"}
_TRUE_VALUES = {"1", "true", "yes", "on"}
_BENCHMARK_POLICY_ENV = "MYCODER_ENABLE_BENCHMARK_POLICY"
_BENCHMARK_IMAGE_RE = re.compile(
    r"^(?:swebench/sweb\.eval\.[A-Za-z0-9._-]+(?::[A-Za-z0-9._-]+)?|mycoder-sandbox:[A-Za-z0-9._-]+)$"
)

# Default price table (USD per 1k tokens) for /v1/agent/cost. Mirrors the
# input/output rates in mycoder/llm.py _PRICING (per 1M -> per 1k).
DEFAULT_PRICE_PER_1K: dict = {
    "gpt-5.5": {"input": 0.005, "output": 0.03},
    "gpt-5.4": {"input": 0.0025, "output": 0.015},
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "deepseek-chat": {"input": 0.00027, "output": 0.0011},
    "deepseek-flash": {"input": 0.00027, "output": 0.0011},
    "claude-sonnet-4-6": {"input": 0.003, "output": 0.015},
    "default": {"input": 0.001, "output": 0.005},
}


# --------------------------------------------------------------------------
# Pydantic v2 request / response models
# --------------------------------------------------------------------------
class RunRequest(BaseModel):
    task: str = Field(min_length=1, description="Task handed to the Orchestrator")
    session_id: str | None = None
    max_tokens: int | None = Field(default=None, ge=1, description="Optional token budget")
    soft_budget_ratio: float | None = Field(
        default=None,
        ge=0.5,
        le=0.95,
        description="Optional convergence threshold as a fraction of max_tokens",
    )
    resume: bool = Field(
        default=False,
        description="Resume this session from its persisted orchestration checkpoint",
    )
    subtasks: list[dict] | None = Field(
        default=None,
        description=(
            "Explicit subagent assignments. When omitted the Orchestrator decomposes the task itself via the TaskPlanner."
        ),
    )
    workspace_id: str = Field(
        default="default",
        pattern=r"^[A-Za-z0-9_.-]{1,64}$",
        description="Tenant-local workspace identifier",
    )
    execution_mode: Literal["single", "multi"] = "multi"
    reasoning_strategy: Literal["auto", "react", "plan_execute", "reflection"] = "auto"
    orchestration_strategy: Literal["auto", "sequential", "parallel", "conditional"] = "auto"
    sandbox_policy: Literal["interactive", "benchmark"] = "interactive"
    # Benchmark adapters may select the official per-instance image.  The
    # server validates the image prefix before it reaches Docker; interactive
    # callers cannot use this field to run arbitrary images.
    sandbox_image: str | None = Field(default=None, max_length=256)
    # None -> the deployment default (MYCODER_SANDBOX_USER, else the image's
    # non-root "sandbox" user). "sandbox" is not hard-coded here so that a
    # Linux host whose uid differs from the image's 1000 can be served by
    # one environment variable instead of a client change.
    sandbox_user: Literal["sandbox", "root"] | None = None
    # P0-4: the harness owns the verification verdict. When this command is set
    # the server runs it inside the sandbox and reads its exit code, instead of
    # trusting that the model ran a check itself. Benchmark-only, exactly like
    # sandbox_image, so an interactive caller cannot ask the server to execute
    # an arbitrary command on its behalf.
    benchmark_verify_cmd: str | None = Field(default=None, max_length=2000)
    benchmark_verify_timeout: int | None = Field(default=None, ge=1, le=3600)


class RunResponse(BaseModel):
    session_id: str
    status: Literal["running"] = "running"


class StatusResponse(BaseModel):
    session_id: str
    status: str
    current_step: str | None = None
    token_usage: int | None = None
    output: str | None = None
    error: dict | None = None
    # Mainstream LLM-service performance metrics, aggregated from the shared
    # LLMTracer after the run: latency (avg/p95 ms), tokens (prompt/completion/
    # total), LLM call count, error calls, cost (USD) and per-model cost.
    perf: dict | None = None
    resumed: bool = False
    checkpoint: dict | None = None
    # The harness-owned verification verdict for a benchmark run: which command
    # ran, whether it exited 0, and its output tail. Present only when a
    # benchmark_verify_cmd / MYCODER_BENCHMARK_VERIFY_CMD was configured — in
    # that case it is what "verified" means for this run. `status` is one of
    # passed / failed / unavailable; `passed` is None when the command could not
    # be run at all, which is not the same fact as a red check.
    harness_verification: dict | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    redis: Literal["connected", "disconnected"]
    version: str


class ErrorResponse(BaseModel):
    code: str
    detail: str
    session_id: str | None = None


class CostResponse(BaseModel):
    session_id: str
    summary: dict
    cost: dict


class ToolTraceResponse(BaseModel):
    session_id: str
    summary: dict
    traces: list[dict]


class MetricsResponse(BaseModel):
    total_runs: int
    completed: int
    running: int
    success: int
    failed: int
    success_rate: float
    failure_rate: float
    failure_distribution: dict


class DeadLetterResponse(BaseModel):
    jobs: list[dict]


class AlertsResponse(BaseModel):
    alerts: list[dict]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_ACTIVE_SESSIONS: set[str] = set()
_JOB_LEASE_SECONDS = max(5, int(os.getenv("MYCODER_JOB_LEASE_SECONDS", "60")))
_JOB_POLL_SECONDS = max(0.05, float(os.getenv("MYCODER_JOB_POLL_SECONDS", "0.5")))
_JOB_MAX_ATTEMPTS = max(1, int(os.getenv("MYCODER_JOB_MAX_ATTEMPTS", "3")))
_JOB_RETRY_BASE_SECONDS = max(0.01, float(os.getenv("MYCODER_JOB_RETRY_BASE_SECONDS", "1")))
_JOB_RETRY_MAX_SECONDS = max(_JOB_RETRY_BASE_SECONDS, float(os.getenv("MYCODER_JOB_RETRY_MAX_SECONDS", "30")))
_SSE_POLL_SECONDS = max(0.05, float(os.getenv("MYCODER_SSE_POLL_SECONDS", "0.25")))
_SSE_HEARTBEAT_SECONDS = max(1.0, float(os.getenv("MYCODER_SSE_HEARTBEAT_SECONDS", "15")))


def sanitize_session_id(session_id: str | None) -> str:
    """Allow only [A-Za-z0-9_-], 1..64 chars — prevents Redis key injection and
    path traversal. Invalid input is scrubbed; an empty one gets a fresh id."""
    sid = (session_id or "").strip()
    if _SESSION_ID_RE.fullmatch(sid):
        return sid
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "", sid)[:64]
    return cleaned or uuid.uuid4().hex[:16]


def _job_retry_delay(attempts: int) -> float:
    """Bounded exponential backoff with jitter to avoid a retry stampede."""
    exponent = min(max(0, int(attempts) - 1), 31)
    cap = min(_JOB_RETRY_MAX_SECONDS, _JOB_RETRY_BASE_SECONDS * (2**exponent))
    return random.uniform(cap / 2, cap)


def resolve_workspace(principal: Principal, workspace_id: str) -> Path:
    """Resolve a tenant-local workspace without permitting symlink escape."""
    base = Path(os.getenv("MYCODER_WORKSPACE_ROOT", os.getcwd())).resolve()
    if principal.key_id == "local-dev" and workspace_id == "default":
        candidate = base
    else:
        candidate = (base / principal.tenant_id / workspace_id).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="workspace escapes configured root") from exc
    if not candidate.is_dir():
        raise HTTPException(status_code=404, detail=f"workspace '{workspace_id}' not found")
    return candidate


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Tear down the process-global sandbox when the server exits.

    The implementer subagent can create a DockerSandbox through the same
    module-global manager the CLI uses; without this, uvicorn's exit would
    leave the container + workspace volume behind. Defensive: None when no
    sandbox tool ever ran.
    """
    from . import dependencies as dependency_module

    backend_provider = _app.dependency_overrides.get(get_state_backend, dependency_module.get_state_backend)
    checkpoint_provider = _app.dependency_overrides.get(get_checkpoint_store, get_checkpoint_store)
    backend = backend_provider()
    checkpoint_store = checkpoint_provider()
    stop = asyncio.Event()
    worker = asyncio.create_task(
        _durable_worker(backend, checkpoint_store, stop),
        name="mycoder-durable-worker",
    )
    yield
    stop.set()
    worker.cancel()
    with suppress(asyncio.CancelledError):
        await worker
    from mycoder.sandbox.executor import get_active_manager

    manager = get_active_manager()
    if manager is not None:
        await manager.stop()
    # Close the state backend (Redis connection) so uvicorn's exit doesn't leak
    # the pool (#14). getattr keeps backends without close (local) a no-op.
    close = getattr(backend, "close", None)
    if close is not None:
        try:
            await close()
        except Exception:  # noqa: BLE001 - teardown must never raise
            pass
    dependency_module.reset_observability_runtime()


app = FastAPI(title="MyCoder Service", version=__version__, lifespan=lifespan)


async def _renew_claim(
    state_backend: StateBackend,
    job_id: str,
    session_id: str,
    worker_id: str,
) -> None:
    """Heartbeat both the queue claim and the per-session execution lease."""
    while True:
        await asyncio.sleep(_JOB_LEASE_SECONDS / 3)
        job_ok = await state_backend.renew_job(job_id, worker_id, _JOB_LEASE_SECONDS)
        session_ok = await state_backend.renew_lease(f"session:{session_id}", worker_id, _JOB_LEASE_SECONDS)
        if not job_ok or not session_ok:
            logger.error("job_lease_lost", job_id=job_id, session_id=session_id)
            return


async def _process_job_id(
    state_backend: StateBackend,
    checkpoint_store: CheckpointStore,
    job_id: str | None = None,
    worker_id: str | None = None,
) -> bool:
    worker_id = worker_id or f"worker-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    job = await state_backend.claim_job(worker_id, _JOB_LEASE_SECONDS, job_id=job_id)
    if job is None:
        return False
    payload = job["payload"]
    session_id = payload["session_id"]
    lease_name = f"session:{session_id}"
    if not await state_backend.acquire_lease(lease_name, worker_id, _JOB_LEASE_SECONDS):
        await state_backend.retry_job(job["job_id"], worker_id, delay_seconds=_job_retry_delay(job["attempts"]))
        return False

    heartbeat = asyncio.create_task(_renew_claim(state_backend, job["job_id"], session_id, worker_id))
    completed = False
    settled = False
    cancelled = False
    failure_reason = "worker interrupted before acknowledgement"
    try:
        existing = await state_backend.get_session(session_id) or {}
        # If a process died after persisting the terminal record but before ACK,
        # the recovered job is acknowledged without replaying side effects.
        if job["attempts"] > 1 and existing.get("status") in ("success", "failed"):
            completed = True
            return True
        # A hard-killed worker cannot execute finally. Its expired claim is
        # recovered with attempts+1; stop before a fourth execution.
        if job["attempts"] > _JOB_MAX_ATTEMPTS:
            failure_reason = f"maximum queue attempts exceeded ({job['attempts'] - 1}/{_JOB_MAX_ATTEMPTS})"
            settled = await state_backend.dead_letter_job(job["job_id"], worker_id, failure_reason)
            if settled:
                await _fail(state_backend, session_id, "JOB_MAX_ATTEMPTS", failure_reason)
            return settled
        recovered = job["attempts"] > 1 and checkpoint_store.load(session_id) is not None
        await _run_agent(
            state_backend,
            session_id,
            payload["task"],
            payload.get("max_tokens"),
            None if (payload.get("resume") or recovered) else payload.get("subtasks"),
            bool(payload.get("resume") or recovered),
            checkpoint_store,
            metadata={
                "tenant_id": payload["tenant_id"],
                "public_session_id": payload["public_session_id"],
                "workspace_id": payload.get("workspace_id", "default"),
                "workspace_root": payload.get("workspace_root"),
                "execution_mode": payload.get("execution_mode", "multi"),
                "reasoning_strategy": payload.get("reasoning_strategy", "auto"),
                "orchestration_strategy": payload.get("orchestration_strategy", "auto"),
                "sandbox_policy": payload.get("sandbox_policy", "interactive"),
                "sandbox_image": payload.get("sandbox_image"),
                "sandbox_user": payload.get("sandbox_user"),
                "benchmark_verify_cmd": payload.get("benchmark_verify_cmd"),
                "benchmark_verify_timeout": payload.get("benchmark_verify_timeout"),
                "soft_budget_ratio": payload.get("soft_budget_ratio"),
            },
        )
        completed = True
        return True
    except asyncio.CancelledError:
        # Cancellation is a deliberate shutdown signal, not a transient
        # worker crash. Retrying here re-queues the same job while the server
        # is stopping and replays it unexpectedly on the next startup.
        cancelled = True
        failure_reason = "worker cancelled during shutdown; job was not replayed"
        raise
    except BaseException as exc:
        failure_reason = f"{type(exc).__name__}: {exc}"[:2000]
        raise
    finally:
        heartbeat.cancel()
        with suppress(asyncio.CancelledError):
            await heartbeat
        if completed:
            await state_backend.complete_job(job["job_id"], worker_id)
        elif cancelled and not settled:
            try:
                settled = await asyncio.shield(
                    state_backend.dead_letter_job(
                        job["job_id"], worker_id, failure_reason
                    )
                )
                if settled:
                    await asyncio.shield(
                        _fail(
                            state_backend,
                            session_id,
                            "TASK_CANCELLED",
                            failure_reason,
                        )
                    )
            except Exception:  # noqa: BLE001 - shutdown cleanup is best-effort
                logger.exception(
                    "cancelled_job_cleanup_failed",
                    job_id=job["job_id"],
                    session_id=session_id,
                )
        elif not settled:
            if job["attempts"] >= _JOB_MAX_ATTEMPTS:
                settled = await state_backend.dead_letter_job(job["job_id"], worker_id, failure_reason)
                if settled:
                    await _fail(
                        state_backend,
                        session_id,
                        "JOB_MAX_ATTEMPTS",
                        failure_reason,
                    )
            else:
                await state_backend.retry_job(
                    job["job_id"],
                    worker_id,
                    delay_seconds=_job_retry_delay(job["attempts"]),
                )
        await state_backend.release_lease(lease_name, worker_id)


async def _durable_worker(
    state_backend: StateBackend,
    checkpoint_store: CheckpointStore,
    stop: asyncio.Event,
) -> None:
    """Continuously recover pending/expired jobs from SQLite or Redis."""
    worker_id = f"worker-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    while not stop.is_set():
        try:
            processed = await _process_job_id(state_backend, checkpoint_store, worker_id=worker_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - worker loop must self-heal
            logger.exception("durable_worker_error", detail=str(exc))
            processed = False
        if not processed:
            try:
                await asyncio.wait_for(stop.wait(), timeout=_JOB_POLL_SECONDS)
            except TimeoutError:
                pass


async def _run_agent(
    state_backend: StateBackend,
    session_id: str,
    task: str,
    max_tokens: int | None,
    subtasks: list[dict] | None = None,
    resume: bool = False,
    checkpoint_store: CheckpointStore | None = None,
    metadata: dict | None = None,
) -> None:
    """Background worker: run the Orchestrator and persist the outcome.

    subtasks is passed through when the client supplied one; when None the
    Orchestrator decomposes the task itself via the TaskPlanner."""
    tracer = get_tracer()
    budget_guard = TokenBudgetGuard(max_tokens_per_session=max_tokens, tracer=tracer)
    bind_contextvars(session_id=session_id)
    manager = None
    run_recorder = None
    try:
        await state_backend.save_session(
            session_id,
            {
                **(metadata or {}),
                "status": "running",
                "current_step": "executing",
                "token_usage": 0,
                "error": None,
                "resumed": resume,
            },
        )
        build = get_orchestrator(state_backend)
        tracer.register_budget_guard(session_id, budget_guard)
        checkpoint_store = checkpoint_store or get_checkpoint_store()
        orchestrator = build(
            session_id,
            budget_guard=budget_guard,
            checkpoint_store=checkpoint_store,
            workspace_root=(metadata or {}).get("workspace_root"),
            reasoning_strategy=(
                None
                if (metadata or {}).get("reasoning_strategy", "auto") == "auto"
                else (metadata or {}).get("reasoning_strategy")
            ),
            sandbox_policy=(metadata or {}).get("sandbox_policy", "interactive"),
            sandbox_image=(metadata or {}).get("sandbox_image"),
            # Keep the default explicit in the queued run contract.  Passing
            # ``None`` through to custom orchestrator builders made the
            # effective default depend on their implementation (and broke
            # audit/replay metadata); benchmark callers can still select
            # ``root`` explicitly after providing an image.
            sandbox_user=(metadata or {}).get("sandbox_user") or "sandbox",
            soft_budget_ratio=(metadata or {}).get("soft_budget_ratio"),
            # Opt-in trace replay: MYCODER_RUN_LOG_DIR turns the run into an
            # append-only event file that `python -m mycoder.replay` can re-execute
            # without a provider.
            run_context={
                "execution_mode": (metadata or {}).get("execution_mode", "multi"),
                "sandbox_policy": (metadata or {}).get("sandbox_policy", "interactive"),
                "workspace_root": (metadata or {}).get("workspace_root"),
            },
        )
        contract = RunContract.from_policy((metadata or {}).get("sandbox_policy", "interactive"))
        # P0-4: with a verification command configured the harness owns the
        # verdict, so the agent-side requirement is dropped instead of being
        # asked for twice. Without one, nothing changes: the agent must still
        # produce its own exit-code evidence (and the fallback path logs a
        # warning saying so).
        if benchmark_verify_command((metadata or {}).get("benchmark_verify_cmd")):
            contract = contract.with_harness_verification()
        manager = getattr(orchestrator, "_sandbox_manager", None)
        run_recorder = getattr(orchestrator, "_run_recorder", None)
        harness_verification: dict | None = None
        if manager is not None:
            # P0-1: the container mounts this very checkout read-write, so
            # "undo" is no longer free — a destructive command hits the real
            # repository. Record where the tree stood before any command runs
            # and keep it with the session, so the working tree stays
            # recoverable and the recovery command survives in the record.
            point = await asyncio.to_thread(manager.capture_restore_point)
            if point is not None and metadata is not None:
                metadata["restore_point"] = point.as_metadata()
        # Benchmark correctness depends on a before/after repository snapshot.
        # Start the sandbox before any host-side write_file call so get_diff()
        # cannot accidentally take its baseline after the edit already landed.
        if (metadata or {}).get("sandbox_policy") == "benchmark" and manager is not None:
            await _start_benchmark_sandbox(manager)
        if (metadata or {}).get("execution_mode", "multi") == "single":
            reasoning_strategy = (
                None
                if (metadata or {}).get("reasoning_strategy", "auto") == "auto"
                else (metadata or {}).get("reasoning_strategy")
            )
            agent_factory = getattr(orchestrator, "agent_factory", None)
            if agent_factory is not None:
                single = agent_factory.build(
                    reasoning_strategy=reasoning_strategy,
                    budget_guard=budget_guard,
                    require_mutation=contract.require_mutation,
                    # A benchmark result is only useful when the model has
                    # exercised the changed behaviour.  Requiring a diff
                    # alone lets a semantically wrong patch pass the API
                    # boundary and defers all feedback to the evaluator.
                    require_verification=contract.require_verification,
                    strict_tool_choice=contract.strict_tool_choice,
                    run_contract=contract,
                    max_turn_tokens=max_tokens,
                    mutation_reserved_tokens=contract.mutation_reserved_tokens,
                    verification_reserved_tokens=contract.verification_reserved_tokens,
                )
            else:
                # Backward-compatible path for injected/custom orchestrators.
                from mycoder.agent import Agent

                single = Agent(
                    llm=orchestrator.llm,
                    tools=orchestrator.tools,
                    reasoning_strategy=reasoning_strategy,
                    budget_guard=budget_guard,
                    require_mutation=contract.require_mutation,
                    require_verification=contract.require_verification,
                    strict_tool_choice=contract.strict_tool_choice,
                    max_turn_tokens=max_tokens,
                    mutation_reserved_tokens=contract.mutation_reserved_tokens,
                    verification_reserved_tokens=contract.verification_reserved_tokens,
                )
            output = await asyncio.to_thread(single.chat, task)
            if (metadata or {}).get("sandbox_policy") == "benchmark":
                harness_verification = await _enforce_benchmark_verification(
                    manager,
                    single,
                    verify_cmd=(metadata or {}).get("benchmark_verify_cmd"),
                    verify_timeout=(metadata or {}).get("benchmark_verify_timeout"),
                )
            token_total = get_tracer().get_session_summary(session_id)["total_tokens"]
            result = SimpleNamespace(
                success=True,
                results={},
                summary=output,
                tokens_used=token_total,
            )
        else:
            result = await orchestrator.orchestrate(
                task=task,
                strategy=OrchestrationStrategy((metadata or {}).get("orchestration_strategy", "auto")),
                subtasks=subtasks,
                parent_context={
                    "session_id": session_id,
                    "task_id": session_id,
                    **contract.as_context(),
                },
                resume=resume,
            )
            # The harness verdict applies to the WORKSPACE, which is the same
            # checkout in both modes — so a configured command must run in multi
            # mode too. Leaving it single-mode-only made the field a silent
            # no-op exactly where the SWE-bench adapter defaults to.
            if (
                (metadata or {}).get("sandbox_policy") == "benchmark"
                and benchmark_verify_command((metadata or {}).get("benchmark_verify_cmd"))
            ):
                harness_verification = await _enforce_benchmark_verification(
                    manager,
                    None,
                    verify_cmd=(metadata or {}).get("benchmark_verify_cmd"),
                    verify_timeout=(metadata or {}).get("benchmark_verify_timeout"),
                )

        # envelope-level failures -> API errors
        budget = max_tokens or budget_guard.max_tokens_per_session
        patch_failures: list[str] = []
        for name, env in result.results.items():
            code = getattr(getattr(env, "error", None), "code", None)
            if code in _BUDGET_CODES:
                raise TokenBudgetExceeded(session_id=session_id, used_tokens=budget, max_tokens=budget)
            if code in _CIRCUIT_BREAKER_CODES:
                raise CircuitBreakerOpen(f"subagent '{name}' skipped by circuit breaker")
            if code in _TIMEOUT_CODES:
                raise SandboxTimeout(f"subagent '{name}' exceeded its hard timeout")
            if code in _PATCH_CODES:
                detail = str(getattr(getattr(env, "error", None), "message", ""))
                patch_failures.append(f"subagent '{name}' [{code}]: {detail[:1200]}")
        if patch_failures:
            raise PatchRequired("; ".join(patch_failures))
        if (metadata or {}).get("sandbox_policy") == "benchmark":
            # The sandbox mounts the project directory itself, so there is one
            # checkout: whatever the agent changed already shows up in
            # `git status` here. No container->host reconciliation step exists
            # any more (that step is what used to silently discard valid
            # patches whose edits had been made inside the container).
            workspace_root = (metadata or {}).get("workspace_root") or (
                manager.project_dir if manager is not None else None
            )
            if workspace_root is not None and not await _workspace_has_repository_changes(workspace_root):
                raise PatchRequired("benchmark run completed without repository changes")
        perf = _session_perf(session_id)
        token_usage = int((perf or {}).get("total_tokens") or result.tokens_used or 0)
        if max_tokens and token_usage > max_tokens:
            raise TokenBudgetExceeded(session_id=session_id, used_tokens=token_usage, max_tokens=max_tokens)

        await state_backend.save_session(
            session_id,
            {
                **(metadata or {}),
                **(
                    {"harness_verification": harness_verification}
                    if harness_verification is not None
                    else {}
                ),
                "status": "success" if result.success else "failed",
                "current_step": "done",
                "token_usage": token_usage,
                "output": result.summary,
                "error": None if result.success else {"code": "SUBAGENT_FAILED", "detail": result.summary[:200]},
                "results": {
                    name: {"status": env.status, "summary": (env.summary or "")[:200]} for name, env in result.results.items()
                },
                "perf": perf,
                "resumed": resume,
            },
        )
        if result.success:
            checkpoint_store.clear(session_id)
        logger.info("agent_run_completed", status="success" if result.success else "failed", tokens=token_usage)
    except TokenBudgetExceeded as exc:
        await _fail(state_backend, session_id, "TOKEN_BUDGET_EXCEEDED", str(exc), perf=_session_perf(session_id))
    except CircuitBreakerOpen as exc:
        await _fail(state_backend, session_id, "CIRCUIT_BREAKER_OPEN", str(exc), perf=_session_perf(session_id))
    except SandboxTimeout as exc:
        await _fail(state_backend, session_id, "SANDBOX_TIMEOUT", str(exc), perf=_session_perf(session_id))
    except ToolChoiceCapabilityError as exc:
        await _fail(state_backend, session_id, "TOOL_CHOICE_UNSUPPORTED", str(exc), perf=_session_perf(session_id))
    except PatchRequired as exc:
        await _fail(state_backend, session_id, "PATCH_REQUIRED", str(exc), perf=_session_perf(session_id))
    except Exception as exc:  # noqa: BLE001 - a worker must never die silently
        await _fail(state_backend, session_id, "INTERNAL_ERROR", str(exc), perf=_session_perf(session_id))
    finally:
        tracer.unregister_budget_guard(session_id, budget_guard)
        _ACTIVE_SESSIONS.discard(session_id)
        if manager is not None:
            try:
                await manager.stop()
            except Exception:  # noqa: BLE001 - sandbox teardown is best-effort
                pass
        if run_recorder is not None:
            try:
                run_recorder.close()
            except Exception:  # noqa: BLE001 - logging must not mask job cleanup
                logger.warning("run_log_close_failed", session_id=session_id)


async def _enforce_benchmark_verification(
    manager,
    agent,
    *,
    verify_cmd: str | None = None,
    verify_timeout: int | None = None,
) -> dict | None:
    """Judge a benchmark run's verification postcondition.

    Two different questions, answered differently on purpose.

    When the harness owns a command — per run via ``benchmark_verify_cmd``, else
    process-wide via ``MYCODER_BENCHMARK_VERIFY_CMD`` — the harness runs it and
    the verdict is that command's EXIT CODE, so the run does not have to be
    trusted on its own word. The verdict is RECORDED, not gated: an unsolved
    instance is data, not a broken run. Rejecting a red check here would turn
    every failing patch into an error case (and re-run it on ``--resume``),
    while the official evaluator is the thing that actually scores the patch.

    With no command configured the run has to certify itself, so the agent's own
    exit-code-based evidence is required — and that stays a gate, because the
    alternative is reporting unverified work as success.

    Three verdicts, not two. ``status`` is ``passed`` / ``failed`` /
    ``unavailable``; ``passed`` is None for the third. A command that could not
    be run at all (an image with no test runtime, a command that is not on
    PATH) says nothing about the patch, so counting it as red — which is what a
    bare ``exit_code != 0`` did — turned "we cannot check this" into "the model
    failed". ``unavailable`` is also not a gate, for the same reason ``failed``
    is not: the official evaluator scores the patch, and re-running every such
    case would just burn the budget.
    """
    command = benchmark_verify_command(verify_cmd)
    if command is not None:
        if manager is None:
            raise PatchRequired(
                "harness verification configured but no sandbox manager is available"
            )
        # Emulation is slower than the host by a wide margin, so a timeout sized
        # for native execution would expire on a check that was progressing
        # normally. The backend reports its own factor (1.0 when native).
        scale = getattr(manager, "timeout_scale", None)
        budget = benchmark_verify_timeout(verify_timeout)
        if callable(scale):
            budget = int(budget * float(scale()))
        outcome = await manager.verify(command, timeout=budget)
        logger.info(
            "api.benchmark_verification",
            verified=outcome.passed,
            status=outcome.status,
            command=command[:160],
        )
        if not outcome.available:
            # The check never ran — no test runtime in the image, no command on
            # PATH. That is "cannot tell", not "the model got it wrong", so it
            # is reported as such and does not count against the run.
            logger.warning(
                "api.benchmark_verification_unavailable",
                command=command[:160],
                hint=(
                    "bake a test runtime into the image, enable "
                    "MYCODER_SANDBOX_NETWORK, or supply a per-instance "
                    "--verify-commands mapping"
                ),
            )
        return outcome.as_dict()
    logger.warning(
        "api.benchmark_verification_self_reported",
        hint=f"set {_BENCHMARK_VERIFY_CMD_ENV} or pass benchmark_verify_cmd",
    )
    evidence = list(getattr(agent, "verification_evidence", []))
    if not evidence or not any(item.get("succeeded") for item in evidence):
        raise PatchRequired(
            "benchmark run completed without a harness verification command and "
            "without a successful verification command of its own"
        )
    return None


async def _start_benchmark_sandbox(manager) -> None:
    """Start the sandbox container up front for a benchmark run.

    ``SandboxManager.get()`` only selects a backend; the container is created
    lazily. Forcing it now keeps the first `execute_in_sandbox` off the
    critical path and surfaces an unavailable sandbox as an explicit error
    before any tokens are spent.
    """
    backend = await manager.get()
    if backend is None:
        raise SandboxTimeout("benchmark sandbox is unavailable")
    await backend.start()


async def _workspace_has_repository_changes(workspace_root: str | Path) -> bool:
    """Return whether a checkout has tracked or untracked host-side changes."""
    process = await asyncio.create_subprocess_exec(
        "git",
        "-C",
        str(Path(workspace_root).resolve()),
        "status",
        "--porcelain=v1",
        "--untracked-files=normal",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _stderr = await process.communicate()
    return process.returncode == 0 and bool(stdout.strip())


async def _fail(
    state_backend: StateBackend,
    session_id: str,
    code: str,
    detail: str,
    perf: dict | None = None,
) -> None:
    previous = await state_backend.get_session(session_id) or {}
    await state_backend.save_session(
        session_id,
        {
            **previous,
            "status": "failed",
            "current_step": "done",
            "token_usage": int((perf or {}).get("total_tokens") or 0),
            "error": {"code": code, "detail": detail},
            "perf": perf,
            "resumed": bool(previous.get("resumed", False)),
        },
    )
    logger.warning("agent_run_failed", error_code=code, detail=detail)


def _session_perf(session_id: str) -> dict | None:
    """Aggregate the run's LLM trace into concrete performance metrics.

    Mirrors what production LLM services report: latency (avg / p95 ms),
    token volume (prompt / completion / total), LLM call count, error calls,
    cost (USD) and a per-model cost breakdown.
    """
    tracer = get_tracer()
    summary = tracer.get_session_summary(session_id)
    tool_summary = get_tool_tracer().get_session_summary(session_id)
    if summary["total_calls"] == 0 and not (tool_summary["calls"] or tool_summary["requirement_misses"]):
        return None
    cost = tracer.get_cost_estimate(session_id, price_per_1k=DEFAULT_PRICE_PER_1K)
    return {
        "llm_calls": summary["total_calls"],
        "prompt_tokens": summary["prompt_tokens"],
        "completion_tokens": summary["completion_tokens"],
        "reasoning_tokens": summary["reasoning_tokens"],
        "total_tokens": summary["total_tokens"],
        "avg_latency_ms": summary["avg_duration_ms"],
        "p95_latency_ms": summary["p95_duration_ms"],
        "error_calls": summary["error_count"],
        "cost_usd": cost["total_cost_usd"],
        "by_model": {m: v["cost"] for m, v in cost["by_model"].items()},
        "tools": {key: value for key, value in tool_summary.items() if key != "session_id"},
    }


def _tenant_sessions(sessions: list[dict], principal: Principal) -> list[dict]:
    # Unscoped historical/local-development records remain visible only in the
    # unauthenticated local tenant; authenticated tenants require exact tags.
    allowed = {principal.tenant_id}
    if principal.key_id == "local-dev":
        allowed.add(None)
    return [item for item in sessions if item.get("tenant_id") in allowed]


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.post("/v1/agent/run", response_model=RunResponse, status_code=202)
async def run(
    body: RunRequest,
    background_tasks: BackgroundTasks,
    state_backend: StateBackend = Depends(get_state_backend),
    checkpoint_store: CheckpointStore = Depends(get_checkpoint_store),
    principal: Principal = Depends(get_principal),
    request: Request = None,  # type: ignore[assignment] - injected by FastAPI
) -> RunResponse:
    public_session_id = sanitize_session_id(body.session_id)
    session_id = scope_session_id(principal.tenant_id, public_session_id)
    bind_contextvars(session_id=session_id)
    limiter = RATE_LIMITER or get_rate_limiter()
    if limiter is not None:
        client = request.client.host if request and request.client else "unknown"
        key = f"{principal.tenant_id}:{client}"
        if not await asyncio.to_thread(limiter.allow, key):
            raise HTTPException(status_code=429, detail="rate limit exceeded")
    if get_default_llm() is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "no LLM configured: set MYCODER_API_KEY, OPENAI_API_KEY, "
                "OPENROUTER_API_KEY, or DEEPSEEK_API_KEY; "
                "MYCODER_PROFILE selects the provider"
            ),
        )
    if body.sandbox_policy == "benchmark":
        enabled = os.getenv(_BENCHMARK_POLICY_ENV, "").strip().lower() in _TRUE_VALUES
        if not enabled or not body.workspace_id.startswith("swe-"):
            raise HTTPException(
                status_code=403,
                detail=(f"benchmark sandbox policy requires an swe-* workspace and {_BENCHMARK_POLICY_ENV}=true"),
            )
        if body.sandbox_image and not _BENCHMARK_IMAGE_RE.fullmatch(body.sandbox_image):
            raise HTTPException(
                status_code=400,
                detail="sandbox_image must be an approved SWE-bench or MyCoder image",
            )
        if body.sandbox_user == "root" and not body.sandbox_image:
            raise HTTPException(
                status_code=400,
                detail="sandbox_user=root requires an explicit benchmark image",
            )
        if body.benchmark_verify_timeout is not None and not body.benchmark_verify_cmd:
            raise HTTPException(
                status_code=400,
                detail="benchmark_verify_timeout requires benchmark_verify_cmd",
            )
        if body.benchmark_verify_cmd and not body.benchmark_verify_cmd.strip():
            raise HTTPException(
                status_code=400,
                detail="benchmark_verify_cmd must not be blank",
            )
        if not benchmark_verify_command(body.benchmark_verify_cmd):
            # Not an error: an interactive run may legitimately have no command.
            # But a benchmark run that falls back to the model's own evidence is
            # scoring itself, so make the weaker path visible in the logs.
            logger.warning(
                "api.benchmark_verification_self_reported",
                session_id=public_session_id,
                hint=f"set {_BENCHMARK_VERIFY_CMD_ENV} or pass benchmark_verify_cmd",
            )
    elif (
        body.sandbox_image
        or body.sandbox_user not in (None, "sandbox")
        or body.benchmark_verify_cmd
        or body.benchmark_verify_timeout is not None
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "sandbox_image, sandbox_user and benchmark_verify_cmd are only "
                "available in benchmark mode"
            ),
        )
    workspace_root = resolve_workspace(principal, body.workspace_id)
    if body.execution_mode == "single" and body.resume:
        raise HTTPException(status_code=400, detail="resume is supported only in multi mode")
    if session_id in _ACTIVE_SESSIONS:
        raise HTTPException(status_code=409, detail=f"session '{public_session_id}' is already running")
    if body.resume:
        checkpoint = checkpoint_store.load(session_id)
        if checkpoint is None:
            raise HTTPException(
                status_code=404,
                detail=f"checkpoint for session '{public_session_id}' not found",
            )
        checkpoint_task = str(checkpoint.get("task") or "")
        if checkpoint_task and checkpoint_task != body.task:
            raise HTTPException(
                status_code=409,
                detail="resume task does not match the checkpointed task",
            )
        previous_session = await state_backend.get_session(session_id) or {}
        previous_workspace = previous_session.get("workspace_id")
        if previous_workspace and previous_workspace != body.workspace_id:
            raise HTTPException(
                status_code=409,
                detail="resume workspace does not match the checkpointed run",
            )
    else:
        # Reusing a session id for a fresh task must never inherit old results.
        checkpoint_store.clear(session_id)
    payload = {
        "session_id": session_id,
        "public_session_id": public_session_id,
        "tenant_id": principal.tenant_id,
        "workspace_id": body.workspace_id,
        "workspace_root": str(workspace_root),
        "execution_mode": body.execution_mode,
        "reasoning_strategy": body.reasoning_strategy,
        "orchestration_strategy": body.orchestration_strategy,
        "sandbox_policy": body.sandbox_policy,
        "sandbox_image": body.sandbox_image,
        "sandbox_user": body.sandbox_user,
        "benchmark_verify_cmd": body.benchmark_verify_cmd,
        "benchmark_verify_timeout": body.benchmark_verify_timeout,
        "soft_budget_ratio": body.soft_budget_ratio,
        "task": body.task,
        "max_tokens": body.max_tokens,
        "resume": body.resume,
        "subtasks": None if body.resume else body.subtasks,
    }
    if not await state_backend.enqueue_job(session_id, payload):
        raise HTTPException(status_code=409, detail=f"session '{public_session_id}' is already queued or running")
    await state_backend.save_session(
        session_id,
        {
            "tenant_id": principal.tenant_id,
            "public_session_id": public_session_id,
            "workspace_id": body.workspace_id,
            "sandbox_policy": body.sandbox_policy,
            "soft_budget_ratio": body.soft_budget_ratio,
            "status": "running",
            "current_step": "queued",
            "token_usage": 0,
            "output": None,
            "error": None,
            "resumed": body.resume,
        },
    )
    _ACTIVE_SESSIONS.add(session_id)
    background_tasks.add_task(
        _process_job_id,
        state_backend,
        checkpoint_store,
        session_id,
    )
    audit(
        "agent_run_scheduled",
        principal,
        session_id=public_session_id,
        workspace_id=body.workspace_id,
    )
    return RunResponse(session_id=public_session_id, status="running")


@app.get("/v1/agent/status/{session_id}", response_model=StatusResponse)
async def status(
    session_id: str,
    state_backend: StateBackend = Depends(get_state_backend),
    checkpoint_store: CheckpointStore = Depends(get_checkpoint_store),
    principal: Principal = Depends(get_principal),
) -> StatusResponse:
    public_sid = sanitize_session_id(session_id)
    sid = scope_session_id(principal.tenant_id, public_sid)
    data = await state_backend.get_session(sid)
    if data is None:
        raise HTTPException(status_code=404, detail=f"session '{public_sid}' not found")
    bind_contextvars(session_id=sid)
    logger.info("agent_status_read", status=data.get("status"))
    audit("agent_status_read", principal, session_id=public_sid)
    return StatusResponse(
        session_id=public_sid,
        status=data.get("status", "unknown"),
        current_step=data.get("current_step"),
        token_usage=data.get("token_usage"),
        output=data.get("output"),
        error=data.get("error"),
        perf=data.get("perf"),
        resumed=bool(data.get("resumed", False)),
        checkpoint=checkpoint_store.summary(sid),
        harness_verification=data.get("harness_verification"),
    )


@app.get("/v1/agent/events/{session_id}")
async def events(
    session_id: str,
    state_backend: StateBackend = Depends(get_state_backend),
    checkpoint_store: CheckpointStore = Depends(get_checkpoint_store),
    principal: Principal = Depends(get_principal),
):
    """Stream durable structured run snapshots as Server-Sent Events.

    Polling ``/status`` remains compatible. This stream observes the same
    backend, so it also works when the run and HTTP connection are handled by
    different workers; checkpoint changes surface as progress events.
    """
    public_sid = sanitize_session_id(session_id)
    sid = scope_session_id(principal.tenant_id, public_sid)
    if await state_backend.get_session(sid) is None:
        raise HTTPException(status_code=404, detail=f"session '{public_sid}' not found")

    async def _stream():
        last_snapshot: str | None = None
        sequence = 0
        last_emit = asyncio.get_running_loop().time()
        yield f"retry: {max(100, int(_SSE_POLL_SECONDS * 1000))}\n\n"
        while True:
            data = await state_backend.get_session(sid)
            if data is None:
                payload = {
                    "session_id": public_sid,
                    "status": "failed",
                    "error": {"code": "SESSION_GONE", "detail": "session record expired"},
                }
                yield f"event: failed\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                return
            payload = {
                "session_id": public_sid,
                "status": data.get("status", "unknown"),
                "current_step": data.get("current_step"),
                "token_usage": data.get("token_usage"),
                "output": data.get("output"),
                "error": data.get("error"),
                "perf": data.get("perf"),
                "resumed": bool(data.get("resumed", False)),
                "checkpoint": checkpoint_store.summary(sid),
            }
            snapshot = json.dumps(payload, ensure_ascii=False, sort_keys=True)
            now = asyncio.get_running_loop().time()
            if snapshot != last_snapshot:
                sequence += 1
                status_name = payload["status"]
                event_name = "completed" if status_name == "success" else "failed" if status_name == "failed" else "progress"
                yield f"id: {sequence}\nevent: {event_name}\ndata: {snapshot}\n\n"
                last_snapshot = snapshot
                last_emit = now
            elif now - last_emit >= _SSE_HEARTBEAT_SECONDS:
                yield ": keep-alive\n\n"
                last_emit = now
            if payload["status"] in ("success", "failed"):
                return
            await asyncio.sleep(_SSE_POLL_SECONDS)

    audit("agent_events_opened", principal, session_id=public_sid)
    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health", response_model=HealthResponse)
async def health(
    state_backend: StateBackend = Depends(get_state_backend),
) -> HealthResponse:
    redis_state: Literal["connected", "disconnected"] = "disconnected"
    if isinstance(state_backend, RedisStateBackend):
        redis_state = "connected" if await state_backend.ping() else "disconnected"
    return HealthResponse(status="ok", redis=redis_state, version=__version__)


@app.get("/v1/agent/metrics", response_model=MetricsResponse)
async def metrics(
    state_backend: StateBackend = Depends(get_state_backend),
    principal: Principal = Depends(get_principal),
) -> MetricsResponse:
    """Production run success-rate aggregation (P2): every recorded session's
    terminal status -> success rate / failure rate / failure distribution."""
    sessions = _tenant_sessions(await state_backend.list_sessions(), principal)
    done = [s for s in sessions if s.get("status") in ("success", "failed")]
    success = sum(1 for s in done if s.get("status") == "success")
    failed = len(done) - success
    dist = Counter((s.get("error") or {}).get("code", "UNKNOWN") for s in done if s.get("status") == "failed")
    return MetricsResponse(
        total_runs=len(sessions),
        completed=len(done),
        running=len(sessions) - len(done),
        success=success,
        failed=failed,
        success_rate=round(success / len(done), 4) if done else 0.0,
        failure_rate=round(failed / len(done), 4) if done else 0.0,
        failure_distribution=dict(dist),
    )


@app.get("/v1/agent/dead-letter", response_model=DeadLetterResponse)
async def dead_letter(
    state_backend: StateBackend = Depends(get_state_backend),
    principal: Principal = Depends(get_principal),
) -> DeadLetterResponse:
    records = await state_backend.list_dead_jobs()
    allowed = {principal.tenant_id}
    if principal.key_id == "local-dev":
        allowed.add(None)
    visible = [record for record in records if (record.get("payload") or {}).get("tenant_id") in allowed]
    audit("agent_dead_letter_read", principal, count=len(visible))
    return DeadLetterResponse(jobs=visible)


@app.get("/v1/agent/alerts", response_model=AlertsResponse)
async def alerts(
    state_backend: StateBackend = Depends(get_state_backend),
    principal: Principal = Depends(get_principal),
) -> AlertsResponse:
    """Newest persisted SLO alerts, tenant-filtered in authenticated mode."""
    records = await asyncio.to_thread(get_alert_manager().list_alerts, 1000)
    if principal.key_id != "local-dev":
        sessions = _tenant_sessions(await state_backend.list_sessions(), principal)
        allowed = {str(session["session_id"]) for session in sessions if session.get("session_id")}
        records = [item for item in records if item.get("session_id") in allowed]
    records = records[:100]
    audit("agent_alerts_read", principal, count=len(records))
    return AlertsResponse(alerts=records)


@app.get("/v1/agent/report")
async def monitor_report(
    state_backend: StateBackend = Depends(get_state_backend),
    principal: Principal = Depends(get_principal),
) -> dict:
    """One monitor snapshot: LLM trace (latency/tokens/TTFT/cost/errors) +
    production run success rate. The "监控报告" endpoint (P3)."""
    from mycoder.observability.report import build_monitor_report

    sessions = _tenant_sessions(await state_backend.list_sessions(), principal)
    trace_session_ids = None
    if principal.key_id != "local-dev":
        trace_session_ids = [str(session["session_id"]) for session in sessions if session.get("session_id")]
    return build_monitor_report(
        get_tracer(),
        sessions=sessions,
        price_per_1k=DEFAULT_PRICE_PER_1K,
        trace_session_ids=trace_session_ids,
        tool_tracer=get_tool_tracer(),
    )


@app.get("/v1/agent/cost/{session_id}", response_model=CostResponse)
async def cost(
    session_id: str,
    state_backend: StateBackend = Depends(get_state_backend),
    principal: Principal = Depends(get_principal),
) -> CostResponse:
    """Per-session summary + cost derived from persisted LLM traces.
    404 when the session has no trace records."""
    public_sid = sanitize_session_id(session_id)
    sid = scope_session_id(principal.tenant_id, public_sid)
    if principal.key_id != "local-dev" and await state_backend.get_session(sid) is None:
        raise HTTPException(status_code=404, detail=f"session '{public_sid}' not found")
    tracer = get_tracer()
    summary = tracer.get_session_summary(sid)
    if summary["total_calls"] == 0:
        raise HTTPException(status_code=404, detail=f"no trace records for session '{public_sid}'")
    cost_estimate = tracer.get_cost_estimate(sid, price_per_1k=DEFAULT_PRICE_PER_1K)
    bind_contextvars(session_id=sid)
    logger.info("agent_cost_read", total_calls=summary["total_calls"], total_tokens=summary["total_tokens"])
    audit("agent_cost_read", principal, session_id=public_sid)
    return CostResponse(session_id=public_sid, summary=summary, cost=cost_estimate)


@app.get(
    "/v1/agent/tool-traces/{session_id}",
    response_model=ToolTraceResponse,
)
async def tool_traces(
    session_id: str,
    limit: int = 200,
    state_backend: StateBackend = Depends(get_state_backend),
    principal: Principal = Depends(get_principal),
) -> ToolTraceResponse:
    """Redacted action timeline for diagnosing tool and mutation failures."""
    public_sid = sanitize_session_id(session_id)
    sid = scope_session_id(principal.tenant_id, public_sid)
    if principal.key_id != "local-dev" and await state_backend.get_session(sid) is None:
        raise HTTPException(status_code=404, detail=f"session '{public_sid}' not found")
    tracer = get_tool_tracer()
    summary = tracer.get_session_summary(sid)
    if not summary["calls"] and not summary["requirement_misses"]:
        raise HTTPException(
            status_code=404,
            detail=f"no tool trace records for session '{public_sid}'",
        )
    bounded_limit = max(1, min(int(limit), 1_000))
    traces = tracer.list_traces(sid, limit=bounded_limit)
    public_summary = {**summary, "session_id": public_sid}
    for trace in traces:
        trace["session_id"] = public_sid
    bind_contextvars(session_id=sid)
    audit(
        "agent_tool_traces_read",
        principal,
        session_id=public_sid,
        count=len(traces),
    )
    return ToolTraceResponse(
        session_id=public_sid,
        summary=public_summary,
        traces=traces,
    )


# --------------------------------------------------------------------------
# Global exception handlers (synchronous setup path; the worker translates its
# own failures into the session record instead).
# --------------------------------------------------------------------------
def _error_response(code: str, detail: str, session_id: str | None = None) -> dict:
    return ErrorResponse(code=code, detail=detail, session_id=session_id).model_dump()


@app.exception_handler(TokenBudgetExceeded)
async def _on_token_budget(_request: Request, exc: TokenBudgetExceeded) -> JSONResponse:
    # 429 = the budget was enforced on a still-running session (rate/budget limit).
    return JSONResponse(status_code=429, content=_error_response("TOKEN_BUDGET_EXCEEDED", str(exc)))


@app.exception_handler(CircuitBreakerOpen)
async def _on_circuit_breaker(_request: Request, exc: CircuitBreakerOpen) -> JSONResponse:
    return JSONResponse(status_code=503, content=_error_response("CIRCUIT_BREAKER_OPEN", str(exc)))


@app.exception_handler(SandboxTimeout)
async def _on_sandbox_timeout(_request: Request, exc: SandboxTimeout) -> JSONResponse:
    return JSONResponse(status_code=504, content=_error_response("SANDBOX_TIMEOUT", str(exc)))
