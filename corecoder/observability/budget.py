"""Token budget enforcement for the LLM layer.

TokenBudgetGuard is the hard trip wire: `check_and_enforce` is called before an
LLM call (SubagentRunner) and after every subtask (Orchestrator). It warns at
80% of the budget and raises TokenBudgetExceeded at 100%. The guard reads a
session's cumulative usage from an attached LLMTracer (the source of truth,
fed by the central LLM.chat instrumentation), falling back to an internal
per-session counter.

max_tokens_per_session defaults from Config (env CORECODER_MAX_TOKENS, 4096).
"""

from __future__ import annotations

from corecoder.config import Config
from corecoder.sandbox.logger import get_logger

logger = get_logger("corecoder.budget")

WARN_RATIO = 0.8


class TokenBudgetExceeded(Exception):
    """Raised when a session's token usage reaches its budget."""

    def __init__(self, session_id: str, used_tokens: int, max_tokens: int) -> None:
        self.session_id = session_id
        self.used_tokens = used_tokens
        self.max_tokens = max_tokens
        super().__init__(
            f"session '{session_id}' exceeded token budget: {used_tokens} >= {max_tokens}"
        )


class TokenBudgetGuard:
    def __init__(
        self,
        max_tokens_per_session: int | None = None,
        tracer=None,
    ) -> None:
        self.max_tokens_per_session = (
            int(max_tokens_per_session)
            if max_tokens_per_session is not None
            else int(Config.from_env().max_tokens)
        )
        self._tracer = tracer
        self._used: dict[str, int] = {}

    def attach_tracer(self, tracer) -> None:
        """Read cumulative usage from an LLMTracer when available."""
        self._tracer = tracer

    def _usage(self, session_id: str) -> int:
        if self._tracer is not None:
            try:
                summary = self._tracer.get_session_summary(session_id)
                if summary["total_calls"] > 0:
                    return int(summary["total_tokens"])
            except Exception:  # noqa: BLE001 - fall back to local counter
                pass
        return self._used.get(session_id, 0)

    def add_usage(self, session_id: str, tokens: int) -> None:
        """Manually accumulate usage (also fed automatically by the tracer)."""
        self._used[session_id] = self._used.get(session_id, 0) + max(0, int(tokens))

    def check_and_enforce(self, session_id: str, current_usage: int | None = None) -> int:
        """Enforce the budget: warn at 80%, raise TokenBudgetExceeded at 100%.

        Returns the remaining token allowance for the session.
        """
        used = (
            int(current_usage)
            if current_usage is not None
            else self._usage(session_id)
        )
        self._used[session_id] = max(self._used.get(session_id, 0), used)

        if used >= self.max_tokens_per_session:
            logger.error(
                "token_budget_exceeded",
                session_id=session_id,
                used_tokens=used,
                max_tokens=self.max_tokens_per_session,
            )
            raise TokenBudgetExceeded(
                session_id=session_id,
                used_tokens=used,
                max_tokens=self.max_tokens_per_session,
            )
        if used >= WARN_RATIO * self.max_tokens_per_session:
            logger.warning(
                "token_budget_warning",
                session_id=session_id,
                used_tokens=used,
                max_tokens=self.max_tokens_per_session,
                remaining=self.max_tokens_per_session - used,
            )
        return self.max_tokens_per_session - used

    def get_remaining(self, session_id: str) -> int:
        return max(0, self.max_tokens_per_session - self._usage(session_id))
