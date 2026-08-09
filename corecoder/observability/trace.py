"""LLM call-level tracing — the reasoning-layer counterpart to MCPCallTrace
(corecoder/mcp/observability.py).

One structured LLMCallTrace per LLM HTTP call, aggregated in process memory.
The `trace()` contextmanager is the integration point: it times the call, lets
the caller fill in token usage, and emits a structlog record. All failures are
recorded (status=error/timeout) and re-raised, never swallowed.

Thread safety: LLM calls may run in worker threads via asyncio.to_thread, so
the trace list is guarded by a lock. tiktoken is an optional dependency — when
missing, estimated token counts are None (recorded as -1) and everything else
keeps working.
"""

from __future__ import annotations

import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator

from corecoder.sandbox.logger import get_logger

logger = get_logger("corecoder.llm_trace")

try:  # optional dependency — estimates degrade to None when absent
    import tiktoken

    _ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:  # noqa: BLE001 - tiktoken is optional
    _ENCODER = None


def estimate_tokens(text: str) -> int | None:
    """Best-effort token count via tiktoken; None when tiktoken is missing."""
    if not text:
        return 0
    if _ENCODER is None:
        return None
    try:
        return len(_ENCODER.encode(text))
    except Exception:  # noqa: BLE001 - tracing must never fail on encode errors
        return None


def _new_id() -> str:
    return uuid.uuid4().hex


def _is_timeout(exc: BaseException) -> bool:
    return isinstance(exc, TimeoutError) or "timeout" in type(exc).__name__.lower()


def _ms(started: float) -> float:
    return round((time.monotonic() - started) * 1000, 2)


@dataclass
class LLMCallTrace:
    """One LLM HTTP call. Field vocabulary mirrors the MCPCallTrace discipline:
    a dataclass that serializes to a structured audit record."""

    call_id: str
    session_id: str
    caller: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    duration_ms: float
    status: str  # success | error | timeout
    error_msg: str | None = None
    timestamp: float = field(default_factory=time.time)


class LLMTracer:
    """In-memory LLM trace store + per-session aggregation.

    Process memory only (resets on restart). One instance can serve many
    sessions; every method is keyed by session_id.
    """

    def __init__(self, budget_guard: Any | None = None) -> None:
        self._traces: list[LLMCallTrace] = []
        self._lock = threading.Lock()
        self.budget_guard = budget_guard

    def attach_budget_guard(self, guard: Any) -> None:
        """Feed completed calls' token usage into a TokenBudgetGuard."""
        self.budget_guard = guard

    # ---------------------------------------------------------------- trace
    @contextmanager
    def trace(self, session_id: str, caller: str, model: str) -> Iterator[dict]:
        """Time one LLM call. Yields a mutable dict the caller fills with
        prompt_tokens / completion_tokens before the block exits. On exception
        the trace is recorded as error/timeout and re-raised."""
        ctx: dict[str, Any] = {"prompt_tokens": 0, "completion_tokens": 0}
        started = time.monotonic()
        try:
            yield ctx
        except Exception as exc:  # noqa: BLE001 - record then re-raise
            status = "timeout" if _is_timeout(exc) else "error"
            self._record(session_id, caller, model, 0, 0, _ms(started), status, str(exc))
            logger.warning(
                "llm_call_failed",
                session_id=session_id,
                caller=caller,
                model=model,
                status=status,
                error_msg=str(exc),
                duration_ms=_ms(started),
            )
            raise

        prompt = int(ctx.get("prompt_tokens") or 0)
        completion = int(ctx.get("completion_tokens") or 0)
        duration = _ms(started)
        self._record(session_id, caller, model, prompt, completion, duration, "success", None)
        logger.info(
            "llm_call",
            session_id=session_id,
            caller=caller,
            model=model,
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
            duration_ms=duration,
        )

    def _record(
        self,
        session_id: str,
        caller: str,
        model: str,
        prompt: int,
        completion: int,
        duration_ms: float,
        status: str,
        error_msg: str | None,
    ) -> None:
        trace = LLMCallTrace(
            call_id=_new_id(),
            session_id=session_id,
            caller=caller,
            model=model,
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
            duration_ms=duration_ms,
            status=status,
            error_msg=error_msg,
        )
        with self._lock:
            self._traces.append(trace)
        if self.budget_guard is not None:
            try:
                self.budget_guard.add_usage(session_id, prompt + completion)
            except Exception:  # noqa: BLE001 - budget is best-effort
                pass

    # -------------------------------------------------------------- summary
    def get_session_summary(self, session_id: str) -> dict:
        """Aggregate counters for one session (total calls, tokens, duration,
        avg/p95, error count)."""
        with self._lock:
            traces = [t for t in self._traces if t.session_id == session_id]
        n = len(traces)
        empty = {
            "session_id": session_id,
            "total_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "total_duration_ms": 0.0,
            "avg_duration_ms": 0.0,
            "p95_duration_ms": 0.0,
            "error_count": 0,
        }
        if n == 0:
            return empty
        prompt = sum(t.prompt_tokens for t in traces)
        completion = sum(t.completion_tokens for t in traces)
        total_duration = sum(t.duration_ms for t in traces)
        durations = sorted(t.duration_ms for t in traces)
        p95 = durations[min(n - 1, int(n * 0.95) - 1)]
        errors = sum(1 for t in traces if t.status != "success")
        return {
            "session_id": session_id,
            "total_calls": n,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "total_duration_ms": round(total_duration, 2),
            "avg_duration_ms": round(total_duration / n, 2),
            "p95_duration_ms": round(p95, 2),
            "error_count": errors,
        }

    def get_cost_estimate(
        self, session_id: str, price_per_1k: dict[str, Any] | None = None
    ) -> dict:
        """Per-model cost estimate in USD.

        price_per_1k shape (USD per 1k tokens):
            {"<model>": {"input": x, "output": y}, "default": {...}}
        A flat number is also accepted ({"<model>": 0.002}). Models without an
        entry fall back to "default"; models with no price contribute 0.
        """
        price_per_1k = price_per_1k or {}
        with self._lock:
            traces = [t for t in self._traces if t.session_id == session_id]
        per_model: dict[str, dict[str, Any]] = {}
        for t in traces:
            bucket = per_model.setdefault(
                t.model, {"calls": 0, "prompt": 0, "completion": 0, "cost": 0.0}
            )
            bucket["calls"] += 1
            bucket["prompt"] += t.prompt_tokens
            bucket["completion"] += t.completion_tokens
        total = 0.0
        breakdown: dict[str, dict[str, Any]] = {}
        for model, b in per_model.items():
            rate = price_per_1k.get(model) or price_per_1k.get("default")
            if isinstance(rate, dict):
                cost = b["prompt"] / 1000 * float(rate.get("input", 0)) + b[
                    "completion"
                ] / 1000 * float(rate.get("output", 0))
            elif isinstance(rate, (int, float)):
                cost = (b["prompt"] + b["completion"]) / 1000 * float(rate)
            else:
                cost = 0.0
            b["cost"] = round(cost, 6)
            total += cost
            breakdown[model] = {"calls": b["calls"], "cost": b["cost"]}
        return {
            "session_id": session_id,
            "total_cost_usd": round(total, 6),
            "by_model": breakdown,
        }


def serialize_trace(trace: LLMCallTrace) -> dict:
    """JSON-safe dict for a trace (used by /cost style endpoints)."""
    return asdict(trace)
