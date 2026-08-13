"""Pluggable reasoning strategies (P1) — ReAct / Plan-and-Execute / Reflection.

The agent loop is structurally identical for all three; what changes is the
system-prompt section telling the model HOW to reason:

  * react        — explicit `Thought:` / `Observation:` lines around each tool
                   call, making the implicit loop explicit;
  * plan_execute — plan first, then execute step by step, re-checking the plan
                   when a result diverges (never blindly follow a stale plan);
  * reflection   — reflect on each result before the next action and self-correct
                   instead of repeating the same failed approach.

Switch with PLANNING_STRATEGY=react|plan_execute|reflection (default react).
"""

from __future__ import annotations

import os

REASONING_STRATEGIES = ("react", "plan_execute", "reflection")
DEFAULT_STRATEGY = "react"

_SECTIONS: dict[str, str] = {
    "react": (
        "\n## Reasoning mode: ReAct (explicit)\n"
        "Before every action, output a single line starting with `Thought:` that "
        "states why you are taking this step. After a tool result, output a short "
        "`Observation:` line. Continue the Thought -> Action -> Observation loop "
        "until the task is complete."
    ),
    "plan_execute": (
        "\n## Reasoning mode: Plan-and-Execute\n"
        "Before acting, output a concise `Plan:` (numbered steps with "
        "dependencies). Execute one step at a time; after each step, compare the "
        "result against the plan and adjust or re-plan if it diverges. Do not "
        "blindly follow the original plan when new information contradicts it."
    ),
    "reflection": (
        "\n## Reasoning mode: Reflection\n"
        "After each action's result, output a short `Reflection:` line evaluating "
        "whether the outcome is as expected. If it is not, correct course before "
        "the next action — do not repeat the same failed approach."
    ),
}


def resolve_strategy(strategy: str | None = None) -> str:
    """Resolve the active strategy from an explicit value or PLANNING_STRATEGY."""
    value = strategy
    if value is None:
        value = os.getenv("PLANNING_STRATEGY", DEFAULT_STRATEGY)
    value = str(value).strip().lower()
    return value if value in REASONING_STRATEGIES else DEFAULT_STRATEGY


def build_reasoning_section(strategy: str | None = None) -> str:
    """Return the system-prompt fragment for the active strategy."""
    return _SECTIONS[resolve_strategy(strategy)]
