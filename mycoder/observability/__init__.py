"""LLM/tool observability: durable traces, budgets, limits and alerts.

* trace.py   — LLMCallTrace + LLMTracer (thread-safe recorder/aggregation).
* tool_trace.py — redacted tool-call and required-action traces.
* run_log.py — append-only per-run event stream (recording side of replay).
* store.py   — process-independent SQLite/Redis trace/rate/alert state.
* budget.py  — TokenBudgetExceeded + TokenBudgetGuard.
"""

from .budget import TokenBudgetExceeded, TokenBudgetGuard
from .run_log import (
    RecordingLLM,
    RunLog,
    RunLogError,
    RunLogRecorder,
    recording_model_factory,
)
from .store import (
    AlertStore,
    ObservabilityStore,
    RateLimitStore,
    RedisObservabilityStore,
    SQLiteObservabilityStore,
    TraceStore,
)
from .trace import LLMCallTrace, LLMTracer, estimate_tokens
from .tool_trace import ToolCallTrace, ToolTracer

__all__ = [
    "LLMCallTrace",
    "LLMTracer",
    "ToolCallTrace",
    "ToolTracer",
    "RunLog",
    "RunLogError",
    "RunLogRecorder",
    "RecordingLLM",
    "recording_model_factory",
    "estimate_tokens",
    "TokenBudgetExceeded",
    "TokenBudgetGuard",
    "SQLiteObservabilityStore",
    "RedisObservabilityStore",
    "TraceStore",
    "RateLimitStore",
    "AlertStore",
    "ObservabilityStore",
]
