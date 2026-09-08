from __future__ import annotations

from mycoder.agent import Agent
from mycoder.convergence import (
    ConvergenceController,
    ConvergenceLimits,
    ToolObservation,
)
from mycoder.llm import LLMResponse, ToolCall
from mycoder.tools.base import Tool


class _ReadTool(Tool):
    name = "read"
    description = "read"
    parameters = {"type": "object", "properties": {}}

    def __init__(self) -> None:
        self.calls = 0

    def execute(self) -> str:
        self.calls += 1
        return "unchanged evidence"


class _LoopingLLM:
    max_tokens = 1024

    def __init__(self) -> None:
        self.calls = 0
        self.finalization_tools = None

    def chat(self, *, tools, **_kwargs):
        self.calls += 1
        if not tools:
            self.finalization_tools = tools
            return LLMResponse(content="final summary")
        return LLMResponse(
            tool_calls=[ToolCall(id=f"call-{self.calls}", name="read", arguments={})]
        )


def _observation(controller, name="read", arguments=None, result="same"):
    admission = controller.admit_tool(name, arguments or {})
    return ToolObservation(admission, name, result)


def test_identical_action_is_blocked_after_limit():
    controller = ConvergenceController(
        ConvergenceLimits(max_identical_tool_calls=2)
    )

    assert controller.admit_tool("read_file", {"file_path": "a.py"}).allowed
    assert controller.admit_tool("read_file", {"file_path": "a.py"}).allowed
    blocked = controller.admit_tool("read_file", {"file_path": "a.py"})

    assert not blocked.allowed
    assert "identical read_file call" in blocked.blocked_reason
    assert controller.tool_calls == 2


def test_repeated_results_trigger_stagnation_but_mutation_resets_it():
    controller = ConvergenceController(
        ConvergenceLimits(max_stagnant_rounds=2)
    )

    assert controller.finish_round([_observation(controller)]) is None
    assert controller.finish_round([_observation(controller)]) is None
    mutation = _observation(
        controller,
        name="edit_file",
        arguments={"old": "a", "new": "b"},
        result="Edited file",
    )
    assert controller.finish_round([mutation]) is None
    assert controller.stagnant_rounds == 0

    first_error = _observation(controller, arguments={"path": "missing-1"}, result="Error: no")
    second_error = _observation(controller, arguments={"path": "missing-2"}, result="Error: no")
    assert controller.finish_round([first_error]) is None
    assert "no new evidence" in controller.finish_round([second_error])


def test_repeated_cached_mutation_does_not_fake_progress():
    controller = ConvergenceController(
        ConvergenceLimits(
            max_identical_tool_calls=2,
            max_stagnant_rounds=2,
        )
    )
    arguments = {"file_path": "a.py", "content": "same"}

    first = _observation(
        controller,
        name="write_file",
        arguments=arguments,
        result="Wrote a.py",
    )
    assert controller.finish_round([first]) is None
    cached = _observation(
        controller,
        name="write_file",
        arguments=arguments,
        result="Wrote a.py",
    )
    assert controller.finish_round([cached]) is None
    blocked = _observation(
        controller,
        name="write_file",
        arguments=arguments,
        result="CONVERGENCE_BLOCKED",
    )
    assert "no new evidence" in controller.finish_round([blocked])


def test_agent_repeated_tool_loop_converges_to_tool_free_summary(monkeypatch):
    monkeypatch.setenv("MYCODER_INJECTION_GUARD", "off")
    llm = _LoopingLLM()
    tool = _ReadTool()
    agent = Agent(
        llm=llm,
        tools=[tool],
        convergence_limits=ConvergenceLimits(
            max_rounds=10,
            max_tool_calls=10,
            max_identical_tool_calls=1,
            max_stagnant_rounds=2,
        ),
    )

    assert agent.chat("read until done") == "final summary"
    assert tool.calls == 1
    assert llm.calls == 4
    assert llm.finalization_tools == []
    assert agent._tool_metrics()["convergence"] == {
        "rounds": 3,
        "tool_calls": 1,
        "stagnant_rounds": 2,
        "stop_reason": "no new evidence or successful state change for 2 consecutive rounds",
    }
    assert any(
        "CONVERGENCE_BLOCKED" in message.get("content", "")
        for message in agent.messages
        if message.get("role") == "tool"
    )


def test_agent_normal_plain_text_completion_is_unchanged(monkeypatch):
    monkeypatch.setenv("MYCODER_INJECTION_GUARD", "off")

    class _DoneLLM:
        def chat(self, **_kwargs):
            return LLMResponse(content="done")

    agent = Agent(llm=_DoneLLM(), tools=[])
    assert agent.chat("answer") == "done"


def test_soft_budget_stops_without_an_unaffordable_final_call(monkeypatch):
    monkeypatch.setenv("MYCODER_INJECTION_GUARD", "off")

    class _Guard:
        max_tokens_per_session = 100

        @staticmethod
        def get_remaining(_session_id):
            return 10

    class _MustNotRun:
        max_tokens = 1024

        def chat(self, **_kwargs):
            raise AssertionError("finalization call should have been reserved")

    from structlog.contextvars import bound_contextvars

    agent = Agent(llm=_MustNotRun(), tools=[], budget_guard=_Guard())
    with bound_contextvars(session_id="budget-test"):
        result = agent.chat("answer")

    assert "soft token budget reached" in result
    assert "final-call budget reserved" in result


def test_projected_next_round_preserves_an_affordable_final_call(monkeypatch):
    monkeypatch.setenv("MYCODER_INJECTION_GUARD", "off")

    class _Guard:
        max_tokens_per_session = 100_000

        @staticmethod
        def get_remaining(_session_id):
            return 20_000

    class _FinalLLM:
        max_tokens = 4096

        def __init__(self):
            self.calls = 0

        def chat(self, *, tools, **_kwargs):
            self.calls += 1
            assert tools == []
            return LLMResponse(content="budget-aware summary")

    from structlog.contextvars import bound_contextvars

    llm = _FinalLLM()
    agent = Agent(llm=llm, tools=[], budget_guard=_Guard())
    with bound_contextvars(session_id="projected-budget-test"):
        result = agent.chat("x" * 30_000)

    assert result == "budget-aware summary"
    assert llm.calls == 1
    assert "soft token budget reached" in agent._last_convergence["stop_reason"]
