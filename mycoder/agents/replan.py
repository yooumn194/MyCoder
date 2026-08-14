"""Dynamic re-planning (P1) — deviation detection + graded recovery.

The orchestrator executes a planner DAG up front; a node whose result deviates
from expectation wastes the remaining plan if execution just carries on. This
module detects three deviation grades and recovers from the lightest
intervention upward (对标 Codex / Claude Code / Hermes):

    HARD_FAIL   (envelope failed)        -> retry once when the error is
                                            retryable, otherwise skip + mark
    SOFT_DRIFT  (partial / low-confidence) -> insert a compensation node into
                                            the execution queue (Claude Code
                                            sub-agent compensation)
    GOAL_DRIFT  (off-target, judge-gated)  -> re-plan the remaining DAG with the
                                            TaskPlanner (Hermes re-decomposition)

Every intervention is reduced to an (reason, strategy, recovered) record so an
injectable experience store (e.g. the cross-session memory DB) can reuse the
playbook next time — no dict, just the detector + the record shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DeviationType(Enum):
    HARD_FAIL = "hard_fail"
    SOFT_DRIFT = "soft_drift"
    GOAL_DRIFT = "goal_drift"


@dataclass(frozen=True)
class Deviation:
    type: DeviationType
    detail: dict


class DeviationDetector:
    """Maps a subagent result envelope onto a deviation grade (or None).

    ``goal_judge`` is an optional callable(envelope, goal) -> bool that flags
    "off-target" results whose envelope is otherwise healthy; when None, the
    GOAL_DRIFT grade is never emitted (cheap default — no extra LLM call).
    """

    def __init__(self, goal_judge=None) -> None:
        self._goal_judge = goal_judge

    def check(self, envelope, goal: str | None = None) -> Deviation | None:
        if envelope is None:
            return None
        if envelope.status == "failed":
            error = envelope.error
            return Deviation(
                DeviationType.HARD_FAIL,
                {
                    "code": error.code if error else "unknown",
                    "retryable": bool(error and error.retryable),
                    "message": (error.message if error else "")[:200],
                },
            )
        if envelope.status == "partial" or (
            envelope.status == "success" and envelope.confidence == "low"
        ):
            return Deviation(
                DeviationType.SOFT_DRIFT,
                {
                    "status": envelope.status,
                    "confidence": envelope.confidence,
                    "completeness": envelope.completeness_ratio,
                },
            )
        if self._goal_judge is not None and goal:
            try:
                if self._goal_judge(envelope, goal):
                    return Deviation(
                        DeviationType.GOAL_DRIFT, {"goal": goal[:80]}
                    )
            except Exception:  # noqa: BLE001 - a failing judge never breaks the loop
                pass
        return None
