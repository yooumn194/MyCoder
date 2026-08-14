"""P1 pluggable reasoning strategies (prompts/reasoning.py + agent wiring)."""

import mycoder.prompt as prompt_mod
from mycoder.agent import Agent
from mycoder.llm import LLM
from mycoder.prompts.reasoning import (
    build_reasoning_section,
    resolve_strategy,
    resolve_strategy_by_task,
)
from mycoder.tools import ALL_TOOLS


def test_resolve_strategy_default_and_env(monkeypatch):
    monkeypatch.delenv("PLANNING_STRATEGY", raising=False)
    assert resolve_strategy() == "react"
    monkeypatch.setenv("PLANNING_STRATEGY", "reflection")
    assert resolve_strategy() == "reflection"
    assert resolve_strategy("plan_execute") == "plan_execute"
    assert resolve_strategy("bogus") == "react"  # invalid falls back


def test_build_sections_have_markers():
    assert "Thought" in build_reasoning_section("react")
    assert "Plan" in build_reasoning_section("plan_execute")
    assert "Reflection" in build_reasoning_section("reflection")


def test_system_prompt_includes_section():
    react = prompt_mod.system_prompt(ALL_TOOLS, reasoning_strategy="react")
    assert "Reasoning mode" in react and "Thought" in react
    plan = prompt_mod.system_prompt(ALL_TOOLS, reasoning_strategy="plan_execute")
    assert "Plan-and-Execute" in plan
    assert "Thought" not in plan  # distinct sections


def test_agent_resolves_strategy():
    agent = Agent(
        llm=LLM.__new__(LLM),
        tools=[ALL_TOOLS[0]],
        reasoning_strategy="plan_execute",
    )
    assert agent.reasoning_strategy == "plan_execute"
    assert "Plan-and-Execute" in agent._system


# ----------------------------------------------------------------- auto/manual
def test_auto_route_by_task():
    assert resolve_strategy_by_task("重构整个模块，设计新的架构") == "plan_execute"
    assert resolve_strategy_by_task("migrate the auth service to a new db") == "plan_execute"
    assert resolve_strategy_by_task("修复登录失败的 bug") == "reflection"
    assert resolve_strategy_by_task("debug the OOM in the worker") == "reflection"
    assert resolve_strategy_by_task("搜索一下这个函数在哪里定义") == "react"
    assert resolve_strategy_by_task(None) == "react"


def test_agent_defaults_to_auto_mode(monkeypatch):
    """No explicit strategy / env -> AUTO mode (per-task routing)."""
    monkeypatch.delenv("PLANNING_STRATEGY", raising=False)
    agent = Agent(llm=LLM.__new__(LLM), tools=[ALL_TOOLS[0]])
    assert agent._strategy_mode == "auto"


def test_agent_manual_override_wins(monkeypatch):
    """Explicit arg OR env -> manual fixed mode (the override takes precedence)."""
    monkeypatch.delenv("PLANNING_STRATEGY", raising=False)
    agent = Agent(llm=LLM.__new__(LLM), tools=[ALL_TOOLS[0]], reasoning_strategy="reflection")
    assert agent._strategy_mode == "manual"
    assert agent.reasoning_strategy == "reflection"

    monkeypatch.setenv("PLANNING_STRATEGY", "plan_execute")
    agent2 = Agent(llm=LLM.__new__(LLM), tools=[ALL_TOOLS[0]])
    assert agent2._strategy_mode == "manual"
    assert agent2.reasoning_strategy == "plan_execute"


def test_set_strategy_manual_then_auto():
    agent = Agent(llm=LLM.__new__(LLM), tools=[ALL_TOOLS[0]])
    assert agent._strategy_mode == "auto"

    msg = agent.set_strategy("reflection")
    assert "reflection" in msg
    assert agent._strategy_mode == "manual"
    assert agent.reasoning_strategy == "reflection"
    assert "Reflection" in agent._system  # system prompt rebuilt

    # switching strategy mid-session rebuilds the system prompt too
    agent.set_strategy("plan_execute")
    assert "Plan-and-Execute" in agent._system
    assert "Thought" not in agent._system

    msg = agent.set_strategy("auto")
    assert agent._strategy_mode == "auto"
    assert "自动" in msg


def test_set_strategy_unknown():
    agent = Agent(llm=LLM.__new__(LLM), tools=[ALL_TOOLS[0]])
    msg = agent.set_strategy("bogus")
    assert "未知策略" in msg
    assert agent._strategy_mode == "auto"


def test_auto_mode_rebuilds_system_per_task(monkeypatch):
    """In auto mode, a task routed to a heavier strategy rebuilds the system
    prompt; a react-routed task stays react."""
    from mycoder.llm import LLMResponse

    monkeypatch.delenv("PLANNING_STRATEGY", raising=False)
    agent = Agent(llm=LLM.__new__(LLM), tools=[ALL_TOOLS[0]])
    agent.llm.chat = lambda messages, tools=None, on_token=None: LLMResponse(content="ok")
    assert agent.reasoning_strategy == "react"

    # a plan-style task switches the strategy + system prompt
    agent.chat("重构模块，设计新架构")
    assert agent.reasoning_strategy == "plan_execute"
    assert "Plan-and-Execute" in agent._system
    assert "重构模块" in agent.messages[0]["content"]  # user msg still appended
