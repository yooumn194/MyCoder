"""P1 pluggable reasoning strategies (prompts/reasoning.py + agent wiring)."""

import corecoder.prompt as prompt_mod
from corecoder.agent import Agent
from corecoder.llm import LLM
from corecoder.prompts.reasoning import build_reasoning_section, resolve_strategy
from corecoder.tools import ALL_TOOLS


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
