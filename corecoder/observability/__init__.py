"""LLM observability: call-level tracing + token-budget enforcement.

  * trace.py   — LLMCallTrace + LLMTracer (in-memory, thread-safe, structlog).
  * budget.py  — TokenBudgetExceeded + TokenBudgetGuard.

No OpenTelemetry / Prometheus: everything aggregates in process memory and is
emitted as structured logs, resets on restart.
"""

from .budget import TokenBudgetExceeded, TokenBudgetGuard
from .trace import LLMCallTrace, LLMTracer, estimate_tokens

__all__ = [
    "LLMCallTrace",
    "LLMTracer",
    "estimate_tokens",
    "TokenBudgetExceeded",
    "TokenBudgetGuard",
]
