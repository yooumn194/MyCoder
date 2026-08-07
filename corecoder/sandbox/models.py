"""Shared data types for the sandbox backends."""

from dataclasses import dataclass


@dataclass(slots=True)
class ExecutionResult:
    """Structured outcome of one sandboxed command.

    A plain dataclass (not an exception) so every terminal state — success,
    timeout, blocked, command failure — flows through the same typed channel
    and can be audited and formatted uniformly by callers.

    `slots=True` shrinks each instance (a command may produce thousands of
    them under heavy agent loops) and the class stays frozen-in-practice.
    """

    exit_code: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    blocked: bool = False
    block_reason: str | None = None
    duration_ms: int | None = None
    container_id: str | None = None

    @property
    def ok(self) -> bool:
        """True only for a clean, on-time, unblocked exit."""
        return self.exit_code == 0 and not self.timed_out and not self.blocked
