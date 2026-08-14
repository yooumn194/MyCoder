"""Failure-mode knowledge base — classify evaluation failures so each one
teaches the next optimization (and the dashboard can alarm on a pattern).

The four patterns map to concrete, actionable fixes.
"""

from enum import Enum


class FailurePattern(Enum):
    TOOL_SELECTION = "tool_selection"        # 该用 LSP 用了 grep
    DELEGATION = "delegation"                # 该用 implementer 用了 explorer
    CONTEXT_LOSS = "context_loss"            # Subagent 遗漏关键信息
    RESOURCE_EXHAUSTION = "resource_exhaustion"  # Token/时间超限


_IMPROVEMENTS = {
    FailurePattern.TOOL_SELECTION: "优化工具描述，增加 ✅/❌ 场景标注",
    FailurePattern.DELEGATION: "调整 Orchestrator Prompt，明确委派决策规则",
    FailurePattern.CONTEXT_LOSS: "优化 Subagent 摘要模板，增加关键信息字段",
    FailurePattern.RESOURCE_EXHAUSTION: "调整 Token 预算或拆分任务",
}


class FailureKnowledgeBase:
    def __init__(self) -> None:
        self._cases: dict[FailurePattern, int] = {}
        self._details: list[dict] = []

    def record_failure(self, case: str, pattern: FailurePattern, details: dict) -> None:
        self._cases[pattern] = self._cases.get(pattern, 0) + 1
        self._details.append({"case": case, "pattern": pattern.value, **details})

    def get_trends(self) -> dict[FailurePattern, int]:
        return dict(self._cases)

    def suggest_improvement(self, pattern: FailurePattern) -> str:
        return _IMPROVEMENTS.get(pattern, "请人工分析")

    def few_shots(self) -> str:
        """沉淀的失败经验 → few-shot 段（发现→修复→预防闭环的「预防」）。

        Every recorded pattern becomes one preventive instruction: the system
        prompt tells the agent which failure modes were seen, how often, and
        what the fix is — so the next run avoids repeating them (对标 Hermes
        的"修复 badcase 的 pattern 沉淀为 few-shot 注入 system prompt").
        """
        if not self._cases:
            return ""
        lines = []
        for pattern, count in sorted(self._cases.items(), key=lambda kv: -kv[1]):
            lines.append(
                f"- 失败模式 `{pattern.value}`（已见 {count} 次）→ 对策：{self.suggest_improvement(pattern)}"
            )
        return "已沉淀的失败经验（避免重蹈覆辙）：\n" + "\n".join(lines)

    def inject_few_shots(self, system_prompt: str) -> str:
        """Return the system prompt with the failure few-shots appended."""
        shots = self.few_shots()
        if not shots:
            return system_prompt
        return f"{system_prompt}\n\n# 失败经验（few-shot）\n{shots}"
