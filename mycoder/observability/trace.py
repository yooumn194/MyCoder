"""LLM call-level tracing — the reasoning-layer counterpart to MCPCallTrace
(mycoder/mcp/observability.py).

One structured LLMCallTrace per LLM HTTP call, optionally backed by a shared
SQLite/Redis store. The `trace()` contextmanager times the call and lets
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
from mycoder.observability.store import TraceStore

from mycoder.sandbox.logger import get_logger

logger = get_logger("mycoder.llm_trace")

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
    ttft_ms: float | None = None  # time-to-first-token (streaming), None if n/a
    cached_tokens: int = 0  # prompt tokens served from the provider's prefix cache
    reasoning_tokens: int = 0  # subset of completion tokens used for reasoning
    timestamp: float = field(default_factory=time.time)
    # Provider wire contract metadata.  These fields are optional so traces
    # written by older workers remain readable after a rolling deployment.
    provider: str = "unknown"
    tool_dialect: str | None = None
    wire_tool_names: list[str] = field(default_factory=list)
    tool_choice_requested: Any = None
    tool_choice_wire: Any = None
    strict_tool_choice: bool = False
    tool_choice_degraded: bool = False
    phase: str | None = None


class LLMTracer:
    """LLM trace recorder + per-session aggregation.

    ``store=None`` preserves the lightweight CLI/test behavior. API wiring
    injects a process-independent ObservabilityStore.
    """

    def __init__(
        self,
        budget_guard: Any | None = None,
        store: TraceStore | None = None,
    ) -> None:
        self._traces: list[LLMCallTrace] = []
        self._lock = threading.Lock()
        self.budget_guard = budget_guard
        self._session_budget_guards: dict[str, Any] = {}
        self.alert_manager = None
        self.store = store

    def attach_budget_guard(self, guard: Any) -> None:
        """Feed completed calls' token usage into a TokenBudgetGuard."""
        self.budget_guard = guard

    def register_budget_guard(self, session_id: str, guard: Any) -> None:
        """Bind a hard budget to one session without affecting other runs."""
        with self._lock:
            self._session_budget_guards[session_id] = guard

    def unregister_budget_guard(self, session_id: str, guard: Any | None = None) -> None:
        """Remove a session guard, optionally only when it is the same object."""
        with self._lock:
            current = self._session_budget_guards.get(session_id)
            if current is not None and (guard is None or current is guard):
                self._session_budget_guards.pop(session_id, None)

    def _budget_guard_for(self, session_id: str):
        with self._lock:
            guard = self._session_budget_guards.get(session_id)
        return guard if guard is not None else self.budget_guard

    def attach_alert_manager(self, manager) -> None:
        """Evaluate SLO alerts (observability/alerts.py) after every call."""
        self.alert_manager = manager

    def _session_metrics(self, session_id: str) -> dict:
        summary = self.get_session_summary(session_id)
        n = summary["total_calls"]
        errors = summary["error_count"]
        budget_ratio = None
        guard = self._budget_guard_for(session_id)
        if guard is not None:
            try:
                budget_ratio = 1.0 - (
                    guard.get_remaining(session_id) / max(1, guard.max_tokens_per_session)
                )
            except Exception:  # noqa: BLE001
                pass
        return {
            "calls": n,
            "success_rate": (n - errors) / n if n else 1.0,
            "p95_duration_ms": summary["p95_duration_ms"],
            "total_tokens": summary["total_tokens"],
            "error_count": errors,
            "budget_ratio": budget_ratio,
        }

    # ---------------------------------------------------------------- trace
    @contextmanager
    def trace(
        self,
        session_id: str,
        caller: str,
        model: str,
        *,
        projected_tokens: int = 0,
        provider: str = "unknown",
        tool_dialect: str | None = None,
        wire_tool_names: list[str] | None = None,
        tool_choice_requested: Any = None,
        tool_choice_wire: Any = None,
        strict_tool_choice: bool = False,
        phase: str | None = None,
    ) -> Iterator[dict]:
        """Time one LLM call. Yields a mutable dict the caller fills with
        prompt_tokens / completion_tokens before the block exits. On exception
        the trace is recorded as error/timeout and re-raised."""
        ctx: dict[str, Any] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,
            "reasoning_tokens": 0,
            "ttft_ms": None,
            "provider": str(provider or "unknown"),
            "tool_dialect": tool_dialect,
            "wire_tool_names": list(wire_tool_names or []),
            "tool_choice_requested": tool_choice_requested,
            "tool_choice_wire": tool_choice_wire,
            "strict_tool_choice": bool(strict_tool_choice),
            "tool_choice_degraded": False,
            "phase": phase,
        }
        guard = self._budget_guard_for(session_id)
        if guard is not None:
            guard.check_and_enforce(session_id)
            if projected_tokens:
                guard.ensure_capacity(session_id, projected_tokens)
        started = time.monotonic()
        try:
            yield ctx
        except Exception as exc:  # noqa: BLE001 - record then re-raise
            status = "timeout" if _is_timeout(exc) else "error"
            self._record(
                session_id,
                caller,
                model,
                0,
                0,
                _ms(started),
                status,
                str(exc),
                ttft_ms=ctx.get("ttft_ms"),
                cached_tokens=ctx.get("cached_tokens"),
                reasoning_tokens=ctx.get("reasoning_tokens"),
                provider=ctx.get("provider", provider),
                tool_dialect=ctx.get("tool_dialect", tool_dialect),
                wire_tool_names=ctx.get("wire_tool_names"),
                tool_choice_requested=ctx.get("tool_choice_requested"),
                tool_choice_wire=ctx.get("tool_choice_wire"),
                strict_tool_choice=ctx.get("strict_tool_choice", strict_tool_choice),
                tool_choice_degraded=ctx.get("tool_choice_degraded", False),
                phase=ctx.get("phase", phase),
            )
            logger.warning(
                "llm_call_failed",
                session_id=session_id,
                caller=caller,
                model=model,
                provider=ctx.get("provider", provider),
                tool_dialect=ctx.get("tool_dialect", tool_dialect),
                phase=ctx.get("phase", phase),
                status=status,
                error_msg=str(exc),
                duration_ms=_ms(started),
            )
            raise

        prompt = int(ctx.get("prompt_tokens") or 0)
        completion = int(ctx.get("completion_tokens") or 0)
        duration = _ms(started)
        self._record(
            session_id,
            caller,
            model,
            prompt,
            completion,
            duration,
            "success",
            None,
            ttft_ms=ctx.get("ttft_ms"),
            cached_tokens=ctx.get("cached_tokens"),
            reasoning_tokens=ctx.get("reasoning_tokens"),
            provider=ctx.get("provider", provider),
            tool_dialect=ctx.get("tool_dialect", tool_dialect),
            wire_tool_names=ctx.get("wire_tool_names"),
            tool_choice_requested=ctx.get("tool_choice_requested"),
            tool_choice_wire=ctx.get("tool_choice_wire"),
            strict_tool_choice=ctx.get("strict_tool_choice", strict_tool_choice),
            tool_choice_degraded=ctx.get("tool_choice_degraded", False),
            phase=ctx.get("phase", phase),
        )
        if guard is not None:
            guard.check_and_enforce(session_id)
        logger.info(
            "llm_call",
            session_id=session_id,
            caller=caller,
            model=model,
            prompt_tokens=prompt,
            completion_tokens=completion,
            reasoning_tokens=int(ctx.get("reasoning_tokens") or 0),
            total_tokens=prompt + completion,
            duration_ms=duration,
            provider=ctx.get("provider", provider),
            tool_dialect=ctx.get("tool_dialect", tool_dialect),
            tool_choice_degraded=bool(ctx.get("tool_choice_degraded", False)),
            phase=ctx.get("phase", phase),
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
        ttft_ms: float | None = None,
        cached_tokens: int = 0,
        reasoning_tokens: int = 0,
        provider: str = "unknown",
        tool_dialect: str | None = None,
        wire_tool_names: list[str] | None = None,
        tool_choice_requested: Any = None,
        tool_choice_wire: Any = None,
        strict_tool_choice: bool = False,
        tool_choice_degraded: bool = False,
        phase: str | None = None,
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
            ttft_ms=ttft_ms,
            cached_tokens=int(cached_tokens or 0),
            reasoning_tokens=int(reasoning_tokens or 0),
            provider=str(provider or "unknown"),
            tool_dialect=tool_dialect,
            wire_tool_names=list(wire_tool_names or []),
            tool_choice_requested=tool_choice_requested,
            tool_choice_wire=tool_choice_wire,
            strict_tool_choice=bool(strict_tool_choice),
            tool_choice_degraded=bool(tool_choice_degraded),
            phase=phase,
        )
        persisted = False
        if self.store is not None:
            try:
                self.store.append_trace(asdict(trace))
                persisted = True
            except Exception as exc:  # noqa: BLE001 - tracing cannot break an LLM call
                logger.warning("trace_store_write_failed", error_msg=str(exc))
        if not persisted:
            # Availability fallback: retain the event in this process if the
            # external store is temporarily unavailable.
            with self._lock:
                self._traces.append(trace)
        if self.budget_guard is not None:
            try:
                self.budget_guard.add_usage(session_id, prompt + completion)
            except Exception:  # noqa: BLE001 - budget is best-effort
                pass
        if self.alert_manager is not None:
            try:
                self.alert_manager.evaluate(session_id, self._session_metrics(session_id))
            except Exception:  # noqa: BLE001 - alerts are best-effort
                pass

    # -------------------------------------------------------------- summary
    def _snapshot(self, session_id: str | None = None) -> list[LLMCallTrace]:
        traces: list[LLMCallTrace] = []
        if self.store is not None:
            try:
                for raw in self.store.list_traces(session_id):
                    try:
                        traces.append(LLMCallTrace(**raw))
                    except (TypeError, ValueError):
                        continue
            except Exception as exc:  # noqa: BLE001 - local fallback remains readable
                logger.warning("trace_store_read_failed", error_msg=str(exc))
        with self._lock:
            fallback = list(self._traces)
        if session_id is not None:
            fallback = [trace for trace in fallback if trace.session_id == session_id]
        traces.extend(fallback)
        traces.sort(key=lambda trace: (trace.timestamp, trace.call_id))
        return traces

    def list_sessions(self) -> list[str]:
        """Every session_id seen so far (for report aggregation)."""
        sessions: set[str] = set()
        if self.store is not None:
            try:
                sessions.update(self.store.list_trace_sessions())
            except Exception as exc:  # noqa: BLE001
                logger.warning("trace_store_sessions_failed", error_msg=str(exc))
        with self._lock:
            sessions.update(trace.session_id for trace in self._traces)
        return sorted(sessions)

    def reset(self) -> None:
        """Drop all recorded traces (primarily comparison/test isolation)."""
        if self.store is not None:
            self.store.reset_traces()
        with self._lock:
            self._traces.clear()

    def get_session_summary(self, session_id: str) -> dict:
        """Aggregate counters for one session (total calls, tokens, duration,
        avg/p95, error count)."""
        return self._summarize_traces(self._snapshot(session_id), session_id)

    def get_global_summary(self) -> dict:
        """Aggregate counters across ALL sessions (monitor-report source)."""
        return self._summarize_traces(self._snapshot(), "<all>")

    def get_sessions_summary(self, session_ids: list[str]) -> dict:
        """Aggregate only an allowed set of sessions (tenant-safe reports)."""
        allowed = set(session_ids)
        traces = [trace for trace in self._snapshot() if trace.session_id in allowed]
        return self._summarize_traces(traces, "<selected>")

    def _summarize_traces(self, traces: list, label: str) -> dict:
        n = len(traces)
        empty = {
            "session_id": label,
            "total_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
            "total_duration_ms": 0.0,
            "avg_duration_ms": 0.0,
            "p95_duration_ms": 0.0,
            "avg_ttft_ms": 0.0,
            "p95_ttft_ms": 0.0,
            "prompt_cache_hit_tokens": 0,
            "prompt_cache_hit_rate": 0.0,
            "error_count": 0,
            "by_provider": {},
            "by_tool_dialect": {},
            "tool_choice_degraded_calls": 0,
            "by_phase": {},
        }
        if n == 0:
            return empty
        prompt = sum(t.prompt_tokens for t in traces)
        completion = sum(t.completion_tokens for t in traces)
        reasoning = sum(t.reasoning_tokens for t in traces)
        total_duration = sum(t.duration_ms for t in traces)
        durations = sorted(t.duration_ms for t in traces)
        p95 = durations[min(n - 1, int(n * 0.95) - 1)]
        errors = sum(1 for t in traces if t.status != "success")
        ttfts = [t.ttft_ms for t in traces if t.ttft_ms is not None]
        avg_ttft = sum(ttfts) / len(ttfts) if ttfts else 0.0
        p95_ttft = sorted(ttfts)[min(len(ttfts) - 1, int(len(ttfts) * 0.95) - 1)] if ttfts else 0.0
        cached = sum(t.cached_tokens for t in traces)
        by_provider: dict[str, dict[str, int]] = {}
        by_dialect: dict[str, int] = {}
        by_phase: dict[str, int] = {}
        for trace in traces:
            provider = str(trace.provider or "unknown")
            bucket = by_provider.setdefault(provider, {"calls": 0, "tokens": 0, "errors": 0})
            bucket["calls"] += 1
            bucket["tokens"] += trace.prompt_tokens + trace.completion_tokens
            bucket["errors"] += int(trace.status != "success")
            dialect = str(trace.tool_dialect or "unknown")
            by_dialect[dialect] = by_dialect.get(dialect, 0) + 1
            phase = str(trace.phase or "unknown")
            by_phase[phase] = by_phase.get(phase, 0) + 1
        return {
            "session_id": label,
            "total_calls": n,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "reasoning_tokens": reasoning,
            "total_tokens": prompt + completion,
            "total_duration_ms": round(total_duration, 2),
            "avg_duration_ms": round(total_duration / n, 2),
            "p95_duration_ms": round(p95, 2),
            "avg_ttft_ms": round(avg_ttft, 2),
            "p95_ttft_ms": round(p95_ttft, 2),
            "prompt_cache_hit_tokens": cached,
            "prompt_cache_hit_rate": round(cached / max(1, prompt), 4),
            "error_count": errors,
            "by_provider": by_provider,
            "by_tool_dialect": by_dialect,
            "tool_choice_degraded_calls": sum(t.tool_choice_degraded for t in traces),
            "by_phase": by_phase,
        }

    def get_cost_estimate(self, session_id: str, price_per_1k: dict[str, Any] | None = None) -> dict:
        """Per-model cost estimate in USD.

        price_per_1k shape (USD per 1k tokens):
            {"<model>": {"input": x, "output": y}, "default": {...}}
        A flat number is also accepted ({"<model>": 0.002}). Models without an
        entry fall back to "default"; models with no price contribute 0.
        """
        price_per_1k = price_per_1k or {}
        traces = self._snapshot(session_id)
        per_model: dict[str, dict[str, Any]] = {}
        for t in traces:
            bucket = per_model.setdefault(t.model, {"calls": 0, "prompt": 0, "completion": 0, "cost": 0.0})
            bucket["calls"] += 1
            bucket["prompt"] += t.prompt_tokens
            bucket["completion"] += t.completion_tokens
        total = 0.0
        breakdown: dict[str, dict[str, Any]] = {}
        for model, b in per_model.items():
            rate = price_per_1k.get(model) or price_per_1k.get("default")
            if isinstance(rate, dict):
                cost = b["prompt"] / 1000 * float(rate.get("input", 0)) + b["completion"] / 1000 * float(rate.get("output", 0))
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
