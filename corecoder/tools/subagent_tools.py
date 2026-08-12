"""spawn_subagent — exposes the orchestrator to the main agent as a tool.

The returned envelope (RFC v1.0.1) is formatted for the LLM: status,
confidence, summary, error category (routed to a correction strategy) and a
compact artifact list.
"""

from ..agents.orchestrator import Orchestrator
from ..contracts import category_to_strategy
from .base import Tool


class SpawnSubagentTool(Tool):
    name = "spawn_subagent"
    idempotent = False  # spawning a subagent has side effects (state, cost)
    description = (
        "Create a Subagent to execute an independent task with isolated "
        "context and its own token budget. Types: explorer (fast read-only "
        "search), planner (design a plan), implementer (make code changes), "
        "reviewer (code review). Returns an RFC v1.0.1 result envelope "
        "(status / summary / error category / artifacts)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "subagent_type": {
                "type": "string",
                "enum": ["explorer", "planner", "implementer", "reviewer"],
                "description": "Subagent 类型",
            },
            "task": {"type": "string", "description": "Subagent 需要完成的具体任务"},
            "blocking": {
                "type": "boolean",
                "description": "是否等待 Subagent 完成（默认 true）",
            },
        },
        "required": ["subagent_type", "task"],
    }

    # set by Agent.__init__ after construction (like AgentTool)
    _parent_agent = None

    def __init__(self, orchestrator: Orchestrator | None = None) -> None:
        self.orchestrator = orchestrator

    def execute(self, subagent_type: str, task: str, blocking: bool = True) -> str:
        orchestrator = self._get_orchestrator()
        if orchestrator is None:
            return "Error: spawn_subagent 未初始化（缺少 orchestrator）"
        envelope = self._run(orchestrator, subagent_type, task, blocking)
        return self._format_result(envelope)

    def _get_orchestrator(self) -> Orchestrator | None:
        if self.orchestrator is None and self._parent_agent is not None:
            from ..agents import Blackboard, Orchestrator

            self.orchestrator = Orchestrator(
                blackboard=Blackboard(),
                llm=self._parent_agent.llm,
                tools=self._parent_agent.tools,
            )
        return self.orchestrator

    def _run(self, orchestrator: Orchestrator, subagent_type: str, task: str, blocking: bool):
        from ..sandbox.executor import run_async

        return run_async(orchestrator.spawn_subagent(subagent_type, task, blocking))

    @staticmethod
    def _format_result(envelope) -> str:
        lines = [f"[Sub-agent {envelope.status} ({envelope.confidence})] {envelope.summary}"]
        if envelope.status == "partial":
            lines.append(f"completeness: {envelope.completeness_ratio}")
        if envelope.error is not None:
            strategy = category_to_strategy(envelope.error.category)
            lines.append(
                f"error[{envelope.error.category} -> {strategy}]: "
                f"{envelope.error.code} — {envelope.error.message}"
            )
        if envelope.artifacts:
            shown = ", ".join(f"{a.action}:{a.path}" for a in envelope.artifacts[:5])
            more = f" ... and {len(envelope.artifacts) - 5} more" if len(envelope.artifacts) > 5 else ""
            lines.append(f"artifacts ({len(envelope.artifacts)}): {shown}{more}")
        return "\n".join(lines)
