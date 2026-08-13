"""Subagent definitions — the catalog the Orchestrator can spawn.

Every subagent reports through the same RFC v1.0.1 envelope; the difference
between roles is the allowed toolset, the model tier, the budget, and the
result.type it is expected to fill.
"""

from dataclasses import dataclass, field


@dataclass
class SubagentDefinition:
    """Full definition of one subagent role (v1.0.1: no output_schema — the
    unified envelope replaces it; the role only varies result.type)."""

    name: str
    description: str
    system_prompt: str
    allowed_tools: list[str] = field(default_factory=list)  # tool whitelist
    model_tier: str | None = None          # overrides Model Router; inherits when None
    max_turns: int = 10
    max_tokens: int = 50000
    # Per-subagent context window (token efficiency, P2). A smaller window makes
    # the sub-agent's ContextManager compress history earlier, so long loops
    # don't re-send an ever-growing transcript on every LLM call.
    max_context_tokens: int = 24000
    timeout_seconds: int = 300
    read_only: bool = False
    retry_on_failure: bool = True


# 内置 Subagent 目录
BUILTIN_SUBAGENTS: dict[str, SubagentDefinition] = {
    "explorer": SubagentDefinition(
        name="explorer",
        description="快速只读代码库探索，用于搜索和定位。返回 ExplorationResult。",
        system_prompt=(
            "You are an explorer subagent. Search and locate code precisely. "
            "Prefer grep_search with file_types filters over broad greps."
        ),
        allowed_tools=["grep_search", "list_files", "read_file"],
        read_only=True,
        model_tier="fast",
        max_turns=5,
        max_context_tokens=16000,
    ),
    "planner": SubagentDefinition(
        name="planner",
        description="设计实现计划，不写代码，只输出方案。返回 PlanResult。",
        system_prompt=(
            "You are a planner subagent. Produce a concrete, ordered plan with "
            "dependencies. Do NOT write code."
        ),
        allowed_tools=["grep_search", "list_files", "read_file", "todo_write"],
        read_only=True,
        model_tier="powerful",
        max_turns=4,
        max_context_tokens=12000,
    ),
    "implementer": SubagentDefinition(
        name="implementer",
        description="执行代码修改，严格按计划实施。返回 ImplementationResult。",
        system_prompt=(
            "You are an implementer subagent. Make the planned code changes and "
            "verify them with tests. Report files changed."
        ),
        allowed_tools=["read_file", "write_file", "execute_in_sandbox", "grep_search"],
        model_tier="standard",
        max_turns=8,
        max_tokens=30000,
        max_context_tokens=32000,
    ),
    "reviewer": SubagentDefinition(
        name="reviewer",
        description="代码审查，检查质量和安全问题。返回 ReviewResult。",
        system_prompt=(
            "You are a reviewer subagent. Review code for correctness, quality "
            "and security. Be specific and structured."
        ),
        allowed_tools=["read_file", "grep_search", "list_files"],
        read_only=True,
        model_tier="standard",
        max_turns=5,
        max_context_tokens=16000,
    ),
}
