"""FastAPI service layer for CoreCoder.

Three endpoints + one background worker:

  * POST /v1/agent/run             — schedule an Orchestrator run and return
                                     immediately (BackgroundTasks, so the
                                     request never blocks on the model).
  * GET  /v1/agent/status/{id}     — read the session record the worker wrote.
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

import re
import uuid
from typing import Literal

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from structlog.contextvars import bind_contextvars

from corecoder import __version__
from corecoder.observability.budget import TokenBudgetExceeded, TokenBudgetGuard
from corecoder.sandbox.logger import get_logger

from .dependencies import (
    get_default_llm,
    get_orchestrator,
    get_state_backend,
    get_tracer,
)
from .state_backend import RedisStateBackend, StateBackend

logger = get_logger("corecoder.api")

# --------------------------------------------------------------------------
# Structured error types raised by the service layer and handled globally.
# (They are API-level exceptions; the underlying modules signal the same
# conditions through envelope error codes, which the worker translates.)
# TokenBudgetExceeded is the shared class from corecoder.observability.budget
# (raised by TokenBudgetGuard); CircuitBreakerOpen / SandboxTimeout are API-level.
# --------------------------------------------------------------------------
class CircuitBreakerOpen(Exception):
    """A subagent was skipped because its circuit breaker is open."""


class SandboxTimeout(Exception):
    """A sandbox / subagent execution exceeded its hard timeout."""


_CIRCUIT_BREAKER_CODES = {"CIRCUIT_BREAKER_OPEN"}
_TIMEOUT_CODES = {"SUBAGENT_TIMEOUT"}
_BUDGET_CODES = {"TOKEN_BUDGET_EXCEEDED"}

# Default price table (USD per 1k tokens) for /v1/agent/cost. Mirrors the
# input/output rates in corecoder/llm.py _PRICING (per 1M -> per 1k).
DEFAULT_PRICE_PER_1K: dict = {
    "gpt-5.5": {"input": 0.005, "output": 0.03},
    "gpt-5.4": {"input": 0.0025, "output": 0.015},
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "deepseek-chat": {"input": 0.00027, "output": 0.0011},
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
    subtasks: list[dict] | None = Field(
        default=None,
        description=(
            "Explicit subagent assignments. When omitted the Orchestrator "
            "decomposes the task itself via the TaskPlanner."
        ),
    )


class RunResponse(BaseModel):
    session_id: str
    status: Literal["running"] = "running"


class StatusResponse(BaseModel):
    session_id: str
    status: str
    current_step: str | None = None
    token_usage: int | None = None
    error: dict | None = None
    # Mainstream LLM-service performance metrics, aggregated from the shared
    # LLMTracer after the run: latency (avg/p95 ms), tokens (prompt/completion/
    # total), LLM call count, error calls, cost (USD) and per-model cost.
    perf: dict | None = None


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


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def sanitize_session_id(session_id: str | None) -> str:
    """Allow only [A-Za-z0-9_-], 1..64 chars — prevents Redis key injection and
    path traversal. Invalid input is scrubbed; an empty one gets a fresh id."""
    sid = (session_id or "").strip()
    if _SESSION_ID_RE.fullmatch(sid):
        return sid
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "", sid)[:64]
    return cleaned or uuid.uuid4().hex[:16]


app = FastAPI(title="CoreCoder Service", version=__version__)


async def _run_agent(
    state_backend: StateBackend,
    session_id: str,
    task: str,
    max_tokens: int | None,
    subtasks: list[dict] | None = None,
) -> None:
    """Background worker: run the Orchestrator and persist the outcome.

    subtasks is passed through when the client supplied one; when None the
    Orchestrator decomposes the task itself via the TaskPlanner."""
    bind_contextvars(session_id=session_id)
    try:
        await state_backend.save_session(
            session_id,
            {"status": "running", "current_step": "executing", "token_usage": 0, "error": None},
        )
        build = get_orchestrator(state_backend)
        # Per-request token budget, fed by the shared tracer (source of truth).
        budget_guard = TokenBudgetGuard(max_tokens_per_session=max_tokens, tracer=get_tracer())
        result = await build(session_id, budget_guard=budget_guard).orchestrate(
            task=task,
            subtasks=subtasks,
            parent_context={"session_id": session_id, "task_id": session_id},
        )

        # envelope-level failures -> API errors
        budget = max_tokens or budget_guard.max_tokens_per_session
        for name, env in result.results.items():
            code = getattr(getattr(env, "error", None), "code", None)
            if code in _BUDGET_CODES:
                raise TokenBudgetExceeded(
                    session_id=session_id, used_tokens=budget, max_tokens=budget
                )
            if code in _CIRCUIT_BREAKER_CODES:
                raise CircuitBreakerOpen(f"subagent '{name}' skipped by circuit breaker")
            if code in _TIMEOUT_CODES:
                raise SandboxTimeout(f"subagent '{name}' exceeded its hard timeout")
        if max_tokens and result.tokens_used > max_tokens:
            raise TokenBudgetExceeded(
                session_id=session_id, used_tokens=result.tokens_used, max_tokens=max_tokens
            )

        await state_backend.save_session(
            session_id,
            {
                "status": "success" if result.success else "failed",
                "current_step": "done",
                "token_usage": result.tokens_used,
                "error": None if result.success else {"code": "SUBAGENT_FAILED", "detail": result.summary[:200]},
                "results": {
                    name: {"status": env.status, "summary": (env.summary or "")[:200]}
                    for name, env in result.results.items()
                },
                "perf": _session_perf(session_id),
            },
        )
        logger.info("agent_run_completed", status="success" if result.success else "failed", tokens=result.tokens_used)
    except TokenBudgetExceeded as exc:
        await _fail(state_backend, session_id, "TOKEN_BUDGET_EXCEEDED", str(exc), perf=_session_perf(session_id))
    except CircuitBreakerOpen as exc:
        await _fail(state_backend, session_id, "CIRCUIT_BREAKER_OPEN", str(exc), perf=_session_perf(session_id))
    except SandboxTimeout as exc:
        await _fail(state_backend, session_id, "SANDBOX_TIMEOUT", str(exc), perf=_session_perf(session_id))
    except Exception as exc:  # noqa: BLE001 - a worker must never die silently
        await _fail(state_backend, session_id, "INTERNAL_ERROR", str(exc), perf=_session_perf(session_id))


async def _fail(
    state_backend: StateBackend,
    session_id: str,
    code: str,
    detail: str,
    perf: dict | None = None,
) -> None:
    await state_backend.save_session(
        session_id,
        {
            "status": "failed",
            "current_step": "done",
            "token_usage": None,
            "error": {"code": code, "detail": detail},
            "perf": perf,
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
    if summary["total_calls"] == 0:
        return None
    cost = tracer.get_cost_estimate(session_id, price_per_1k=DEFAULT_PRICE_PER_1K)
    return {
        "llm_calls": summary["total_calls"],
        "prompt_tokens": summary["prompt_tokens"],
        "completion_tokens": summary["completion_tokens"],
        "total_tokens": summary["total_tokens"],
        "avg_latency_ms": summary["avg_duration_ms"],
        "p95_latency_ms": summary["p95_duration_ms"],
        "error_calls": summary["error_count"],
        "cost_usd": cost["total_cost_usd"],
        "by_model": {m: v["cost"] for m, v in cost["by_model"].items()},
    }


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.post("/v1/agent/run", response_model=RunResponse, status_code=202)
async def run(
    body: RunRequest,
    background_tasks: BackgroundTasks,
    state_backend: StateBackend = Depends(get_state_backend),
) -> RunResponse:
    session_id = sanitize_session_id(body.session_id)
    bind_contextvars(session_id=session_id)
    if get_default_llm() is None:
        raise HTTPException(
            status_code=503,
            detail="no LLM configured: set CORECODER_API_KEY (or OPENAI_API_KEY)",
        )
    await state_backend.save_session(
        session_id,
        {"status": "running", "current_step": "queued", "token_usage": 0, "error": None},
    )
    background_tasks.add_task(
        _run_agent, state_backend, session_id, body.task, body.max_tokens, body.subtasks
    )
    logger.info("agent_run_scheduled", task=body.task[:80])
    return RunResponse(session_id=session_id, status="running")


@app.get("/v1/agent/status/{session_id}", response_model=StatusResponse)
async def status(
    session_id: str,
    state_backend: StateBackend = Depends(get_state_backend),
) -> StatusResponse:
    sid = sanitize_session_id(session_id)
    data = await state_backend.get_session(sid)
    if data is None:
        raise HTTPException(status_code=404, detail=f"session '{sid}' not found")
    bind_contextvars(session_id=sid)
    logger.info("agent_status_read", status=data.get("status"))
    return StatusResponse(
        session_id=sid,
        status=data.get("status", "unknown"),
        current_step=data.get("current_step"),
        token_usage=data.get("token_usage"),
        error=data.get("error"),
        perf=data.get("perf"),
    )


@app.get("/health", response_model=HealthResponse)
async def health(
    state_backend: StateBackend = Depends(get_state_backend),
) -> HealthResponse:
    redis_state: Literal["connected", "disconnected"] = "disconnected"
    if isinstance(state_backend, RedisStateBackend):
        redis_state = "connected" if await state_backend.ping() else "disconnected"
    return HealthResponse(status="ok", redis=redis_state, version=__version__)


@app.get("/v1/agent/cost/{session_id}", response_model=CostResponse)
async def cost(session_id: str) -> CostResponse:
    """Per-session LLM trace summary + cost estimate (in-memory, resets on
    restart). 404 when the session has no trace records."""
    sid = sanitize_session_id(session_id)
    tracer = get_tracer()
    summary = tracer.get_session_summary(sid)
    if summary["total_calls"] == 0:
        raise HTTPException(status_code=404, detail=f"no trace records for session '{sid}'")
    cost_estimate = tracer.get_cost_estimate(sid, price_per_1k=DEFAULT_PRICE_PER_1K)
    bind_contextvars(session_id=sid)
    logger.info("agent_cost_read", total_calls=summary["total_calls"], total_tokens=summary["total_tokens"])
    return CostResponse(session_id=sid, summary=summary, cost=cost_estimate)


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
