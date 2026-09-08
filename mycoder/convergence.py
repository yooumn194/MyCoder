"""Deterministic convergence controls for the agent tool loop.

Prompts can encourage a model to stop, but runtime limits are the reliable
boundary. This module keeps the policy independent from tool execution and
provider code so every reasoning strategy gets the same safeguards.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any


_MUTATING_TOOLS = {
    "edit_file",
    "write_file",
    "sync_workspace",
    "todo_write",
    "todo_update",
    "memory_save",
    "memory_forget",
    "memory_confirm",
    "memory_correct",
}


def _positive_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


@dataclass(frozen=True)
class ConvergenceLimits:
    """Bounds for one user turn, configurable without changing code."""

    max_rounds: int = 16
    max_tool_calls: int = 32
    max_identical_tool_calls: int = 2
    max_stagnant_rounds: int = 3
    soft_budget_ratio: float = 0.85

    @classmethod
    def from_env(cls, agent_max_rounds: int) -> "ConvergenceLimits":
        try:
            ratio = float(os.getenv("MYCODER_CONVERGENCE_SOFT_BUDGET_RATIO", "0.85"))
        except ValueError:
            ratio = 0.85
        return cls(
            max_rounds=min(
                max(1, agent_max_rounds),
                _positive_env("MYCODER_CONVERGENCE_MAX_ROUNDS", 16),
            ),
            max_tool_calls=_positive_env("MYCODER_CONVERGENCE_MAX_TOOL_CALLS", 32),
            max_identical_tool_calls=_positive_env(
                "MYCODER_CONVERGENCE_MAX_IDENTICAL_CALLS", 2
            ),
            max_stagnant_rounds=_positive_env(
                "MYCODER_CONVERGENCE_MAX_STAGNANT_ROUNDS", 3
            ),
            soft_budget_ratio=min(0.95, max(0.5, ratio)),
        )


@dataclass(frozen=True)
class ToolAdmission:
    """Decision made once for a model-requested tool call."""

    signature: str
    novel: bool
    blocked_reason: str | None = None

    @property
    def allowed(self) -> bool:
        return self.blocked_reason is None


@dataclass(frozen=True)
class ToolObservation:
    admission: ToolAdmission
    tool_name: str
    result: str


class ConvergenceController:
    """Track progress and decide when the tool loop must hand off to a summary."""

    def __init__(self, limits: ConvergenceLimits) -> None:
        self.limits = limits
        self.rounds = 0
        self.tool_calls = 0
        self.stagnant_rounds = 0
        self._action_counts: dict[str, int] = {}
        self._result_digests: set[str] = set()

    @staticmethod
    def _signature(name: str, arguments: dict[str, Any]) -> str:
        encoded = json.dumps(
            arguments,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return f"{name}:{encoded}"

    @staticmethod
    def _digest(result: str) -> str:
        normalized = " ".join(str(result).split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @staticmethod
    def _succeeded(result: str) -> bool:
        if getattr(result, "status", None) == "error":
            return False
        text = str(result).lstrip()
        return not text.startswith(
            (
                "Error",
                "⚠ Blocked",
                "⚠ Cancelled",
                "CONVERGENCE_BLOCKED",
                "[interrupted]",
            )
        )

    def before_round(self, budget_ratio: float | None = None) -> str | None:
        if self.rounds >= self.limits.max_rounds:
            return f"round limit reached ({self.limits.max_rounds})"
        if budget_ratio is not None and budget_ratio >= self.limits.soft_budget_ratio:
            return f"soft token budget reached ({budget_ratio:.0%})"
        self.rounds += 1
        return None

    def admit_tool(self, name: str, arguments: dict[str, Any]) -> ToolAdmission:
        signature = self._signature(name, arguments)
        count = self._action_counts.get(signature, 0)
        self._action_counts[signature] = count + 1

        if self.tool_calls >= self.limits.max_tool_calls:
            return ToolAdmission(
                signature,
                novel=False,
                blocked_reason=(
                    "CONVERGENCE_BLOCKED: tool-call limit reached; use existing "
                    "evidence and finish the task."
                ),
            )
        if count >= self.limits.max_identical_tool_calls:
            return ToolAdmission(
                signature,
                novel=False,
                blocked_reason=(
                    f"CONVERGENCE_BLOCKED: identical {name} call already ran "
                    f"{count} times; do not repeat it. Change approach or finish."
                ),
            )

        self.tool_calls += 1
        return ToolAdmission(signature, novel=count == 0)

    def finish_round(self, observations: list[ToolObservation]) -> str | None:
        """Return a stop reason after consecutive rounds without new evidence."""
        progressed = False
        for observation in observations:
            admission = observation.admission
            result = observation.result
            if not admission.allowed or not self._succeeded(result):
                continue

            digest = self._digest(result)
            result_is_new = digest not in self._result_digests
            self._result_digests.add(digest)
            if admission.novel and (
                observation.tool_name in _MUTATING_TOOLS or result_is_new
            ):
                progressed = True

        self.stagnant_rounds = 0 if progressed else self.stagnant_rounds + 1
        if self.stagnant_rounds >= self.limits.max_stagnant_rounds:
            return (
                "no new evidence or successful state change for "
                f"{self.stagnant_rounds} consecutive rounds"
            )
        if self.tool_calls >= self.limits.max_tool_calls:
            return f"tool-call limit reached ({self.limits.max_tool_calls})"
        return None

    def snapshot(self, stop_reason: str | None = None) -> dict[str, Any]:
        return {
            "rounds": self.rounds,
            "tool_calls": self.tool_calls,
            "stagnant_rounds": self.stagnant_rounds,
            "stop_reason": stop_reason,
        }
