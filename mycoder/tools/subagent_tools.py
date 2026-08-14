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
    # P1 re-planning experience hook (injected by Agent); persists every
    # (deviation, strategy, recovered) so the playbook is reusable next time.
    _experience_store = None
    # P2 token budget (injected by Agent): per-subagent budget protection.
    _budget_guard = None
    # The orchestrator this tool lazily built is bound to ONE parent Agent.
    # ALL_TOOLS shares a single SpawnSubagentTool across every Agent, so a
    # cached orchestrator must not leak one Agent's llm/budget/experience into
    # another — we rebuild when the parent changed.
    _orchestrator_parent = None
    _orchestrator_injected = False

    def __init__(
        self,
        orchestrator: Orchestrator | None = None,
        experience_store=None,
        budget_guard=None,
    ) -> None:
        self.orchestrator = orchestrator
        self._orchestrator_injected = orchestrator is not None
        if experience_store is not None:
            self._experience_store = experience_store
        if budget_guard is not None:
            self._budget_guard = budget_guard

    def execute(self, subagent_type: str, task: str, blocking: bool = True) -> str:
        orchestrator = self._get_orchestrator()
        if orchestrator is None:
            return "Error: spawn_subagent 未初始化（缺少 orchestrator）"
        envelope = self._run(orchestrator, subagent_type, task, blocking)
        return self._format_result(envelope)

    def _get_orchestrator(self) -> Orchestrator | None:
        parent = self._parent_agent
        # An explicitly injected orchestrator is used as-is; never rebuild it.
        if self._orchestrator_injected:
            return self.orchestrator
        if parent is None:
            return self.orchestrator
        # Rebuild when none exists yet OR the cached one belongs to a different
        # Agent (the ALL_TOOLS singleton is shared across Agents — see
        # _orchestrator_parent above).
        if self.orchestrator is None or self._orchestrator_parent is not parent:
            from ..agents import Blackboard, Orchestrator
            from ..agents.planner import TaskPlanner
            from ..model_router import build_model_factory

            self.orchestrator = Orchestrator(
                blackboard=Blackboard(),
                llm=parent.llm,
                tools=parent.tools,
                # P1 dynamic re-planning: without a planner the orchestrator
                # can only single-explorer decompose, so GOAL_DRIFT re-planning
                # never fires — only retry / compensation would.
                planner=TaskPlanner(llm=parent.llm),
                # P2 model-tier routing (cost): sub-agents get a tier-appropriate
                # model (explorer=fast, planner=powerful, ...) instead of the
                # shared LLM, so simple roles stop paying flagship prices.
                model_factory=build_model_factory(parent.llm),
                # P1 re-planning experience: persist (deviation, strategy,
                # recovered) records so the CLI's memory DB can reuse them.
                experience_store=self._experience_store,
                # P2 token budget: per-subagent budget protection in the CLI.
                budget_guard=self._budget_guard,
            )
            self._orchestrator_parent = parent
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
