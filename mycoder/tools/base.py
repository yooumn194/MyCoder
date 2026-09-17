"""Base class for all tools."""

from __future__ import annotations

from abc import ABC, abstractmethod


class ToolResult(str):
    """A string tool result that also carries structured metadata.

    Being a ``str`` subclass keeps every call site that treats a result as text
    working unchanged, while letting the harness read facts the *prose* can only
    approximate. ``exit_code`` in particular is what turns "did this check
    pass?" from a regex guess over a command line into a fact reported by the
    process that ran it.
    """

    status: str = "success"
    exit_code: int | None = None
    duration_ms: float = 0.0
    retry_count: int = 0
    cache_hit: bool = False

    def __new__(
        cls,
        value,
        *,
        status: str = "success",
        exit_code: int | None = None,
        duration_ms: float = 0.0,
        retry_count: int = 0,
        cache_hit: bool = False,
    ):
        instance = super().__new__(cls, str(value))
        instance.status = status
        instance.exit_code = exit_code
        instance.duration_ms = max(0.0, float(duration_ms))
        instance.retry_count = max(0, int(retry_count))
        instance.cache_hit = bool(cache_hit)
        return instance


class Tool(ABC):
    """Minimal tool interface. Subclass this to add new capabilities."""

    name: str
    description: str
    parameters: dict  # JSON Schema for the function args

    # Retry contract (P0, tools/idempotency.py):
    #   True  -> transient failures may be auto-retried because repeating the
    #            operation cannot double-apply a side effect.
    #   False -> side-effecting or non-deterministic. Never auto-retried.
    idempotent: bool = True

    # Result-cache contract. This is deliberately independent of idempotency:
    # a read is safe to retry but its answer becomes stale when the workspace
    # changes. The conservative default therefore executes every call. Only
    # operations with a stable result independent of mutable external state
    # opt in explicitly.
    cacheable: bool = False

    # Predictive execution starts before the model response has finished. It is
    # therefore stricter than idempotency: only tools that are explicitly
    # read-only and safe to run speculatively may opt in. The conservative
    # default prevents a partially generated tool call from mutating state.
    predictive_safe: bool = False

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Run the tool and return a text result."""
        ...

    def schema(self) -> dict:
        """OpenAI function-calling schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
