"""编排 token 效率优化: 子 agent 上下文窗口/轮次/预算 + planner 简单任务约束."""

import asyncio
import json
from types import SimpleNamespace

from mycoder.agents.definition import BUILTIN_SUBAGENTS, SubagentDefinition
from mycoder.agents.planner_prompt import build_system_prompt
from mycoder.agents.runner import SubagentRunner
from mycoder.tools import ALL_TOOLS

# 真实评测发现: 简单任务烧满 60k token 被预算熔断。以下收紧是让每个子 agent
# 更早压缩历史、更少轮次、更小 token 预算。


def test_definition_has_context_window_default():
    assert SubagentDefinition(name="x", description="d", system_prompt="p").max_context_tokens == 24000


def test_builtin_budgets_tightened():
    assert BUILTIN_SUBAGENTS["implementer"].max_turns <= 8
    assert BUILTIN_SUBAGENTS["implementer"].max_tokens <= 30000
    assert BUILTIN_SUBAGENTS["implementer"].max_context_tokens < 60000
    for name in ("explorer", "planner", "reviewer"):
        assert BUILTIN_SUBAGENTS[name].max_turns <= 8
        assert BUILTIN_SUBAGENTS[name].max_context_tokens <= 24000


def test_subagent_uses_definition_context_window(monkeypatch):
    captured = {}

    class _FakeAgent:
        def __init__(self, **kw):
            captured.update(kw)

        def chat(self, prompt):
            return json.dumps(
                {
                    "meta": {
                        "task_id": "x",
                        "subagent_name": "implementer",
                        "subagent_instance_id": "i",
                        "started_at": "2026-01-01T00:00:00Z",
                        "finished_at": "2026-01-01T00:00:00Z",
                        "duration_ms": 1,
                    },
                    "status": "success",
                    "summary": "ok",
                    "confidence": "high",
                    "result": {"type": "general", "output": "ok"},
                }
            )

    monkeypatch.setattr("mycoder.agent.Agent", _FakeAgent)
    orch = SimpleNamespace(llm=object(), tools=ALL_TOOLS)
    runner = SubagentRunner(
        definition=BUILTIN_SUBAGENTS["implementer"],
        task="t",
        orchestrator=orch,
        parent_context={"task_id": "x"},
    )
    asyncio.run(runner._run_sub_agent("sys"))
    assert captured["max_context_tokens"] == BUILTIN_SUBAGENTS["implementer"].max_context_tokens
    assert captured["max_rounds"] == BUILTIN_SUBAGENTS["implementer"].max_turns


def test_planner_prompt_asks_simple_tasks_not_to_split():
    prompt = build_system_prompt(5)
    assert "简单任务" in prompt
    assert "implementer" in prompt


def test_subagent_uses_model_factory_tier(monkeypatch):
    captured = {}

    class _FakeAgent:
        def __init__(self, **kw):
            captured.update(kw)

        def chat(self, prompt):
            return json.dumps(
                {
                    "meta": {
                        "task_id": "x",
                        "subagent_name": "explorer",
                        "subagent_instance_id": "i",
                        "started_at": "2026-01-01T00:00:00Z",
                        "finished_at": "2026-01-01T00:00:00Z",
                        "duration_ms": 1,
                    },
                    "status": "success",
                    "summary": "ok",
                    "confidence": "high",
                    "result": {"type": "general", "output": "ok"},
                }
            )

    class _Factory:
        def __call__(self, tier):
            return f"llm-for-{tier}"

    monkeypatch.setattr("mycoder.agent.Agent", _FakeAgent)
    orch = SimpleNamespace(
        llm="shared", tools=ALL_TOOLS, model_factory=_Factory()
    )
    runner = SubagentRunner(
        definition=BUILTIN_SUBAGENTS["explorer"],  # model_tier = fast
        task="t",
        orchestrator=orch,
        parent_context={"task_id": "x"},
    )
    asyncio.run(runner._run_sub_agent("sys"))
    assert captured["llm"] == "llm-for-fast"  # explorer routes to the fast tier
