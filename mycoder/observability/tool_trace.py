"""Durable tool-call tracing for agent execution.

LLM traces explain provider latency and token use, but they cannot explain why
an agent failed to change a repository.  ToolTracer records the action layer:
which tool was selected, whether it ran, whether it mutated or verified state,
and a bounded, redacted preview of the outcome.

The existing ObservabilityStore is intentionally reused.  Records are tagged
with ``trace_type=tool`` so LLMTracer ignores them while both SQLite and Redis
retain one process-independent session timeline.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from mycoder.sandbox.logger import get_logger

from .store import TraceStore

logger = get_logger("mycoder.tool_trace")

_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "password",
    "secret",
    "token",
}
_ARGUMENT_LIMIT = 2_000
_RESULT_LIMIT = 1_000


def _redact(text: str) -> str:
    """Load the tool-layer redactor lazily to keep observability acyclic.

    Importing ``mycoder.tools.security`` at module load time executes the tools
    package initializer, which imports subagents and ultimately LLM again.
    """
    from mycoder.tools.security import redact_output

    return redact_output(text)


def _safe_value(value: Any, *, key: str = "") -> Any:
    """Return a JSON-safe, bounded value without persisting obvious secrets."""
    normalized_key = key.lower().replace("-", "_")
    if normalized_key in _SENSITIVE_KEYS or normalized_key.endswith(
        ("_api_key", "_password", "_secret", "_token", "_private_key")
    ):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(k): _safe_value(v, key=str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_value(item) for item in value[:50]]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _redact(str(value))[:_ARGUMENT_LIMIT]


def _safe_arguments(arguments: dict[str, Any] | None) -> dict[str, Any]:
    safe = _safe_value(arguments or {})
    encoded = json.dumps(safe, ensure_ascii=False, default=str)
    if len(encoded) <= _ARGUMENT_LIMIT:
        return safe
    return {"_truncated": _redact(encoded[:_ARGUMENT_LIMIT])}


@dataclass
class ToolCallTrace:
    call_id: str
    session_id: str
    tool_call_id: str
    tool_name: str
    status: str  # success | error | blocked | cache_hit | missing
    duration_ms: float
    arguments: dict[str, Any]
    result_preview: str
    subagent_name: str = "main"
    subagent_instance_id: str | None = None
    retry_count: int = 0
    cache_hit: bool = False
    mutation: bool = False
    verification: bool = False
    event_type: str = "tool_call"  # tool_call | requirement_feedback
    attempt: int | None = None
    phase: str | None = None
    timestamp: float = field(default_factory=time.time)
    trace_type: str = "tool"


class ToolTracer:
    """Thread-safe recorder and per-session tool execution aggregator."""

    def __init__(self, store: TraceStore | None = None) -> None:
        self.store = store
        self._traces: list[ToolCallTrace] = []
        self._lock = threading.Lock()

    def record(
        self,
        session_id: str,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, Any] | None,
        result: Any,
        *,
        status: str,
        duration_ms: float = 0.0,
        subagent_name: str = "main",
        subagent_instance_id: str | None = None,
        retry_count: int = 0,
        cache_hit: bool = False,
        mutation: bool = False,
        verification: bool = False,
        phase: str | None = None,
    ) -> ToolCallTrace:
        trace = ToolCallTrace(
            call_id=uuid.uuid4().hex,
            session_id=str(session_id or "unknown"),
            tool_call_id=str(tool_call_id or ""),
            tool_name=str(tool_name or "unknown"),
            status=status,
            duration_ms=round(max(0.0, float(duration_ms)), 2),
            arguments=_safe_arguments(arguments),
            result_preview=_redact(str(result or ""))[:_RESULT_LIMIT],
            subagent_name=str(subagent_name or "main"),
            subagent_instance_id=subagent_instance_id,
            retry_count=max(0, int(retry_count)),
            cache_hit=bool(cache_hit),
            mutation=bool(mutation),
            verification=bool(verification),
            phase=str(phase) if phase else None,
        )
        self._append(trace)
        logger.info(
            "tool_call",
            session_id=trace.session_id,
            subagent_name=trace.subagent_name,
            tool_name=trace.tool_name,
            status=trace.status,
            duration_ms=trace.duration_ms,
            retry_count=trace.retry_count,
            mutation=trace.mutation,
            verification=trace.verification,
            phase=trace.phase,
        )
        return trace

    def record_requirement_feedback(
        self,
        session_id: str,
        feedback: str,
        *,
        attempt: int,
        phase: str,
        subagent_name: str = "main",
        subagent_instance_id: str | None = None,
    ) -> ToolCallTrace:
        trace = ToolCallTrace(
            call_id=uuid.uuid4().hex,
            session_id=str(session_id or "unknown"),
            tool_call_id=f"requirement-{attempt}",
            tool_name="__mutation_requirement__",
            status="missing",
            duration_ms=0.0,
            arguments={},
            result_preview=_redact(feedback)[:_RESULT_LIMIT],
            subagent_name=str(subagent_name or "main"),
            subagent_instance_id=subagent_instance_id,
            event_type="requirement_feedback",
            attempt=max(1, int(attempt)),
            phase=phase,
        )
        self._append(trace)
        logger.warning(
            "tool_requirement_missing",
            session_id=trace.session_id,
            subagent_name=trace.subagent_name,
            attempt=trace.attempt,
            phase=trace.phase,
        )
        return trace

    def _append(self, trace: ToolCallTrace) -> None:
        persisted = False
        if self.store is not None:
            try:
                self.store.append_trace(asdict(trace))
                persisted = True
            except Exception as exc:  # noqa: BLE001 - tracing never breaks tools
                logger.warning("tool_trace_store_write_failed", error_msg=str(exc))
        if not persisted:
            with self._lock:
                self._traces.append(trace)

    def _snapshot(self, session_id: str | None = None) -> list[ToolCallTrace]:
        traces: list[ToolCallTrace] = []
        if self.store is not None:
            try:
                for raw in self.store.list_traces(session_id):
                    if raw.get("trace_type") != "tool":
                        continue
                    try:
                        traces.append(ToolCallTrace(**raw))
                    except (TypeError, ValueError):
                        continue
            except Exception as exc:  # noqa: BLE001 - local fallback remains usable
                logger.warning("tool_trace_store_read_failed", error_msg=str(exc))
        with self._lock:
            fallback = list(self._traces)
        if session_id is not None:
            fallback = [trace for trace in fallback if trace.session_id == session_id]
        traces.extend(fallback)
        traces.sort(key=lambda trace: (trace.timestamp, trace.call_id))
        return traces

    def list_traces(self, session_id: str, *, limit: int = 200) -> list[dict]:
        traces = self._snapshot(session_id)
        return [asdict(trace) for trace in traces[-max(1, min(int(limit), 1_000)) :]]

    def get_session_summary(self, session_id: str) -> dict[str, Any]:
        return self._summarize(self._snapshot(session_id), session_id)

    def get_global_summary(self) -> dict[str, Any]:
        return self._summarize(self._snapshot(), "<all>")

    def get_sessions_summary(self, session_ids: list[str]) -> dict[str, Any]:
        allowed = set(session_ids)
        return self._summarize(
            [trace for trace in self._snapshot() if trace.session_id in allowed],
            "<selected>",
        )

    def list_sessions(self) -> list[str]:
        return sorted({trace.session_id for trace in self._snapshot()})

    @staticmethod
    def _summarize(traces: list[ToolCallTrace], session_id: str) -> dict[str, Any]:
        calls = [trace for trace in traces if trace.event_type == "tool_call"]
        durations = sorted(trace.duration_ms for trace in calls)
        failures = sum(trace.status in {"error", "blocked"} for trace in calls)
        successes = sum(trace.status in {"success", "cache_hit"} for trace in calls)
        return {
            "session_id": session_id,
            "calls": len(calls),
            "successes": successes,
            "failures": failures,
            "cache_hits": sum(trace.cache_hit for trace in calls),
            "retries": sum(trace.retry_count for trace in calls),
            "mutations": sum(trace.mutation for trace in calls),
            "verifications": sum(trace.verification for trace in calls),
            "requirement_misses": sum(trace.event_type == "requirement_feedback" for trace in traces),
            "success_rate": round(successes / len(calls), 4) if calls else 0.0,
            "avg_duration_ms": round(sum(durations) / len(durations), 2) if durations else 0.0,
            "p95_duration_ms": round(
                durations[min(len(durations) - 1, max(0, int(len(durations) * 0.95) - 1))],
                2,
            )
            if durations
            else 0.0,
        }

    def reset(self) -> None:
        """Clear only the process-local fallback; shared reset belongs to the store."""
        with self._lock:
            self._traces.clear()
