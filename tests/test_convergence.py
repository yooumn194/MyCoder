from __future__ import annotations

from mycoder.agent import Agent
from mycoder.convergence import (
    ConvergenceController,
    ConvergencePhase,
    ConvergenceLimits,
    ToolObservation,
)
from mycoder.llm import LLMResponse, ToolCall
from mycoder.tools.base import Tool, ToolResult
from mycoder.observability.tool_trace import ToolTracer


def test_convergence_phase_transitions_are_monotonic():
    controller = ConvergenceController(ConvergenceLimits())
    assert controller.transition(ConvergencePhase.MUTATE) is True
    assert controller.transition(ConvergencePhase.VERIFY) is True
    # A provider/model cannot reopen exploration after verification starts.
    assert controller.transition(ConvergencePhase.EXPLORE) is False
    assert controller.phase is ConvergencePhase.VERIFY
    assert controller.phase_snapshot()["phase_transitions"][-1]["reason"] == "regression_ignored"


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
        return LLMResponse(tool_calls=[ToolCall(id=f"call-{self.calls}", name="read", arguments={})])


def _observation(controller, name="read", arguments=None, result="same"):
    admission = controller.admit_tool(name, arguments or {})
    return ToolObservation(admission, name, result)


def test_identical_action_is_blocked_after_limit():
    controller = ConvergenceController(ConvergenceLimits(max_identical_tool_calls=2))

    assert controller.admit_tool("read_file", {"file_path": "a.py"}).allowed
    assert controller.admit_tool("read_file", {"file_path": "a.py"}).allowed
    blocked = controller.admit_tool("read_file", {"file_path": "a.py"})

    assert not blocked.allowed
    assert "identical read_file call" in blocked.blocked_reason
    assert controller.tool_calls == 2


def test_repeated_results_trigger_stagnation_but_mutation_resets_it():
    controller = ConvergenceController(ConvergenceLimits(max_stagnant_rounds=2))

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
    assert any("CONVERGENCE_BLOCKED" in message.get("content", "") for message in agent.messages if message.get("role") == "tool")


def test_agent_normal_plain_text_completion_is_unchanged(monkeypatch):
    monkeypatch.setenv("MYCODER_INJECTION_GUARD", "off")

    class _DoneLLM:
        def chat(self, **_kwargs):
            return LLMResponse(content="done")

    agent = Agent(llm=_DoneLLM(), tools=[])
    assert agent.chat("answer") == "done"


def test_required_mutation_rejects_analysis_only_then_allows_write(monkeypatch):
    monkeypatch.setenv("MYCODER_INJECTION_GUARD", "off")

    class _WriteTool(Tool):
        name = "write_file"
        description = "write"
        parameters = {"type": "object", "properties": {}}

        def __init__(self):
            self.calls = 0

        def execute(self):
            self.calls += 1
            return "Wrote 1 line"

    class _LLM:
        def __init__(self):
            self.calls = 0

        def chat(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(content="I found the fix")
            if self.calls == 2:
                return LLMResponse(tool_calls=[ToolCall(id="write", name="write_file", arguments={})])
            return LLMResponse(content="implemented")

    llm = _LLM()
    tool = _WriteTool()
    agent = Agent(llm=llm, tools=[tool], require_mutation=True)

    assert agent.chat("fix") == "implemented"
    assert llm.calls == 3
    assert tool.calls == 1
    assert any("mutation_required" in (message.get("content") or "").lower() for message in agent.messages)


def test_required_mutation_does_not_interrupt_natural_inspection_loop(monkeypatch):
    monkeypatch.setenv("MYCODER_INJECTION_GUARD", "off")

    class _Inspect(Tool):
        name = "read_file"
        description = "inspect"
        parameters = {"type": "object", "properties": {"file_path": {"type": "string"}}}

        def execute(self, file_path):
            return f"contents of {file_path}"

    class _Edit(Tool):
        name = "edit_file"
        description = "edit"
        parameters = {"type": "object", "properties": {}}

        def execute(self):
            return "Edited app.py"

    class _LLM:
        def __init__(self):
            self.calls = 0

        def chat(self, **kwargs):
            self.calls += 1
            assert "tool_choice" not in kwargs
            if self.calls <= 2:
                return LLMResponse(
                    tool_calls=[
                        ToolCall(
                            id=f"read-{self.calls}",
                            name="read_file",
                            arguments={"file_path": f"file-{self.calls}.py"},
                        )
                    ]
                )
            if self.calls == 3:
                return LLMResponse(tool_calls=[ToolCall(id="edit", name="edit_file", arguments={})])
            return LLMResponse(content="implemented")

    tracer = ToolTracer()
    agent = Agent(
        llm=_LLM(),
        tools=[_Inspect(), _Edit()],
        require_mutation=True,
        tool_tracer=tracer,
        max_inspection_rounds_before_action=3,
    )

    assert agent.chat("fix") == "implemented"
    assert not any(
        "required_action_feedback" in str(message.get("content") or "")
        for message in agent.messages
        if message.get("role") == "user"
    )
    assert [event["name"] for event in agent.tool_events] == [
        "read_file",
        "read_file",
        "edit_file",
    ]


def test_required_mutation_bounds_exploration_with_transient_action_budget(monkeypatch):
    monkeypatch.setenv("MYCODER_INJECTION_GUARD", "off")

    class _Inspect(Tool):
        name = "read_file"
        description = "inspect"
        parameters = {"type": "object", "properties": {"file_path": {"type": "string"}}}

        def execute(self, file_path):
            return f"VALUE = 1  # {file_path}"

    class _Edit(Tool):
        name = "edit_file"
        description = "edit"
        parameters = {"type": "object", "properties": {}}

        def execute(self):
            return "Edited app.py"

    class _LLM:
        def __init__(self):
            self.requests = []

        def chat(self, **kwargs):
            self.requests.append(kwargs)
            call = len(self.requests)
            names = {schema["function"]["name"] for schema in kwargs["tools"]}
            system = kwargs["messages"][0]["content"]
            if call <= 2:
                assert "tool_choice" not in kwargs
                assert "<action_budget>" not in system
                return LLMResponse(
                    tool_calls=[
                        ToolCall(id=f"read-{call}", name="read_file", arguments={"file_path": f"file-{call}.py"})
                    ]
                )
            if call == 3:
                assert "tool_choice" not in kwargs
                assert names == {"read_file", "edit_file"}
                assert "<action_budget>" in system
                return LLMResponse(
                    tool_calls=[ToolCall(id="last-read", name="read_file", arguments={"file_path": "file-3.py"})]
                )
            if call == 4:
                # P0-2: the catalog is stable — every tool stays advertised on
                # every round, including the terminal action boundary.
                assert names == {"read_file", "edit_file"}
                assert "tool_choice" not in kwargs
                assert "Call edit_file now" in system
                return LLMResponse(
                    tool_calls=[ToolCall(id="provider-ignored-choice", name="read_file", arguments={"file_path": "file-4.py"})]
                )
            if call == 5:
                assert names == {"read_file", "edit_file"}
                assert "tool_choice" not in kwargs
                return LLMResponse(tool_calls=[ToolCall(id="edit", name="edit_file", arguments={})])
            assert "tool_choice" not in kwargs
            assert "<action_budget>" not in system
            return LLMResponse(content="implemented")

    llm = _LLM()
    agent = Agent(
        llm=llm,
        tools=[_Inspect(), _Edit()],
        require_mutation=True,
        max_inspection_rounds_before_action=2,
    )

    assert agent.chat("fix") == "implemented"
    assert len(llm.requests) == 6
    assert not any("<action_budget>" in str(message.get("content") or "") for message in agent.messages)
    assert [event["name"] for event in agent.tool_events] == [
        "read_file",
        "read_file",
        "read_file",
        "read_file",
        "edit_file",
    ]
    assert agent.tool_events[3]["status"] == "blocked"
    assert agent.tool_events[3]["succeeded"] is False


def test_requirement_phases_never_use_provider_tool_choice(monkeypatch):
    """P0-2: requirements are enforced by the gate + feedback, not tool_choice.

    Forcing ``tool_choice`` is provider-specific (DeepSeek's thinking endpoint
    rejects it outright), so a run's outcome depended on which model it was
    routed to. The catalog is now advertised identically every round and the
    provider is never asked to constrain its choice.
    """
    monkeypatch.setenv("MYCODER_INJECTION_GUARD", "off")

    from mycoder.tool_protocol import ToolProtocolAdapter

    class _LLM:
        tool_protocol = ToolProtocolAdapter.for_model(
            "deepseek-flash", provider="deepseek"
        )

    agent = Agent(llm=_LLM(), tools=[])
    assert not hasattr(agent, "_requirement_tool_policy")


def test_force_edit_recovery_read_is_bounded_and_not_named_edit(monkeypatch):
    """A FILE_NOT_READ retry can inspect once before the next forced edit."""
    monkeypatch.setenv("MYCODER_INJECTION_GUARD", "off")

    from mycoder.tool_protocol import ToolProtocolAdapter

    class _Read(Tool):
        name = "read_file"
        description = "read"
        parameters = {
            "type": "object",
            "properties": {"file_path": {"type": "string"}},
        }

        def __init__(self):
            self.calls = 0

        def execute(self, file_path):
            self.calls += 1
            return f"contents of {file_path}"

    class _Edit(Tool):
        name = "edit_file"
        description = "edit"
        parameters = {"type": "object", "properties": {}}

        def __init__(self):
            self.calls = 0

        def execute(self):
            self.calls += 1
            if self.calls == 1:
                return "Error [FILE_NOT_READ]: target must be read before it can be edited"
            return "Edited app.py"

    class _LLM:
        max_tokens = 1024
        tool_protocol = ToolProtocolAdapter.for_model(
            "deepseek-flash", provider="deepseek"
        )

        def __init__(self):
            self.requests = []

        def chat(self, **kwargs):
            self.requests.append(kwargs)
            call = len(self.requests)
            if call == 1:
                return LLMResponse(
                    tool_calls=[
                        ToolCall(
                            id="read-1",
                            name="read_file",
                            arguments={"file_path": "app.py"},
                        )
                    ]
                )
            if call == 2:
                return LLMResponse(
                    tool_calls=[
                        ToolCall(
                            id="read-2",
                            name="read_file",
                            arguments={"file_path": "app.py"},
                        )
                    ]
                )
            if call == 3:
                return LLMResponse(
                    tool_calls=[ToolCall(id="edit-1", name="edit_file", arguments={})]
                )
            if call == 4:
                names = {schema["function"]["name"] for schema in kwargs["tools"]}
                assert names == {"read_file", "edit_file"}
                # DeepSeek's thinking dialect cannot accept a named choice;
                # leaving it unset makes the targeted recovery read reachable.
                assert "tool_choice" not in kwargs
                return LLMResponse(
                    tool_calls=[
                        ToolCall(
                            id="read-3",
                            name="read_file",
                            arguments={"file_path": "app.py"},
                        )
                    ]
                )
            if call == 5:
                names = {schema["function"]["name"] for schema in kwargs["tools"]}
                # P0-2: the catalog is stable across every round, so the model
                # always sees both tools and can choose the edit itself.
                assert names == {"read_file", "edit_file"}
                assert "tool_choice" not in kwargs
                return LLMResponse(
                    tool_calls=[ToolCall(id="edit-2", name="edit_file", arguments={})]
                )
            return LLMResponse(content="implemented")

    llm = _LLM()
    read = _Read()
    edit = _Edit()
    agent = Agent(
        llm=llm,
        tools=[read, edit],
        require_mutation=True,
        max_inspection_rounds_before_action=1,
    )

    assert agent.chat("fix") == "implemented"
    # The repeated pre-action read may be rejected by convergence accounting;
    # the important invariant is that the recovery read executes at most once.
    assert read.calls <= 2
    assert edit.calls == 2
    assert len(llm.requests) == 6


def test_required_mutation_survives_tool_selection(monkeypatch):
    monkeypatch.setenv("MYCODER_INJECTION_GUARD", "off")

    class _Selector:
        @staticmethod
        def select(_query, tools):
            return [tool for tool in tools if tool.name == "read_file"]

    class _Tool(Tool):
        description = "tool"
        parameters = {"type": "object", "properties": {}}

        def __init__(self, name):
            self.name = name

        def execute(self):
            return "ok"

    class _LLM:
        def chat(self, *, tools, **_kwargs):
            names = {schema["function"]["name"] for schema in tools}
            assert names == {"read_file", "edit_file", "write_file"}
            return LLMResponse(content="done")

    agent = Agent(
        llm=_LLM(),
        tools=[_Tool("read_file"), _Tool("edit_file"), _Tool("write_file")],
        tool_selector=_Selector(),
        require_mutation=True,
        max_mutation_feedback_rounds=1,
    )

    assert "mutation feedback limit reached" in agent.chat("fix")


def test_required_mutation_forces_inspection_then_edit_and_traces_feedback(monkeypatch):
    from structlog.contextvars import bind_contextvars, clear_contextvars

    monkeypatch.setenv("MYCODER_INJECTION_GUARD", "off")

    class _Inspect(Tool):
        name = "read_file"
        description = "inspect"
        parameters = {"type": "object", "properties": {}}

        def execute(self):
            return "VALUE = 1"

    class _Edit(Tool):
        name = "edit_file"
        description = "edit"
        parameters = {"type": "object", "properties": {}}

        def execute(self):
            return "Edited production.py"

    class _LLM:
        max_tokens = 1024

        def __init__(self):
            self.requests = []

        def chat(self, **kwargs):
            self.requests.append(kwargs)
            call = len(self.requests)
            if call == 1:
                return LLMResponse(content="the fix should change VALUE")
            if call == 4:
                assert "tool_choice" not in kwargs
                return LLMResponse(content="implemented")
            names = {schema["function"]["name"] for schema in kwargs.get("tools", [])}
            assert "tool_choice" not in kwargs
            if call == 2:
                assert names == {"read_file", "edit_file"}
                return LLMResponse(tool_calls=[ToolCall(id="inspect", name="read_file", arguments={})])
            if call == 3:
                assert names == {"read_file", "edit_file"}
                return LLMResponse(tool_calls=[ToolCall(id="mutate", name="edit_file", arguments={})])
            raise AssertionError(f"unexpected call {call}")

    tracer = ToolTracer()
    bind_contextvars(session_id="forced-mutation")
    try:
        llm = _LLM()
        agent = Agent(
            llm=llm,
            tools=[_Inspect(), _Edit()],
            require_mutation=True,
            tool_tracer=tracer,
            trace_context={"subagent_name": "implementer", "subagent_instance_id": "i-1"},
        )
        assert agent.chat("fix it") == "implemented"
    finally:
        clear_contextvars()

    summary = tracer.get_session_summary("forced-mutation")
    assert summary["calls"] == 2
    assert summary["mutations"] == 1
    assert summary["requirement_misses"] == 1
    assert [event["name"] for event in agent.tool_events] == [
        "read_file",
        "edit_file",
    ]


def test_required_mutation_feedback_is_bounded(monkeypatch):
    monkeypatch.setenv("MYCODER_INJECTION_GUARD", "off")

    class _NeverActs:
        max_tokens = 1024

        def __init__(self):
            self.requests = []

        def chat(self, **kwargs):
            self.requests.append(kwargs)
            return LLMResponse(content="analysis only")

    class _Inspect(Tool):
        name = "read_file"
        description = "inspect"
        parameters = {"type": "object", "properties": {}}

        def execute(self):
            return "unused"

    llm = _NeverActs()
    agent = Agent(
        llm=llm,
        tools=[_Inspect()],
        require_mutation=True,
        max_mutation_feedback_rounds=2,
    )

    result = agent.chat("fix it")

    assert "mutation feedback limit reached (2)" in result
    assert len(llm.requests) == 2
    assert "tool_choice" not in llm.requests[1]


def test_failed_tool_results_are_not_cached_before_mutation_retry(monkeypatch):
    monkeypatch.setenv("MYCODER_INJECTION_GUARD", "off")

    class _FlakyEdit(Tool):
        name = "edit_file"
        description = "edit"
        parameters = {"type": "object", "properties": {}}

        def __init__(self):
            self.calls = 0

        def execute(self):
            self.calls += 1
            return "Error: stale file" if self.calls == 1 else "Edited production.py"

    class _LLM:
        max_tokens = 1024

        def __init__(self):
            self.calls = 0

        def chat(self, **kwargs):
            self.calls += 1
            if self.calls in {1, 2}:
                return LLMResponse(tool_calls=[ToolCall(id=f"edit-{self.calls}", name="edit_file", arguments={})])
            return LLMResponse(content="implemented")

    tool = _FlakyEdit()
    agent = Agent(llm=_LLM(), tools=[tool], require_mutation=True)

    assert agent.chat("fix it") == "implemented"
    assert tool.calls == 2
    tool_outputs = [message["content"] for message in agent.messages if message.get("role") == "tool"]
    assert any("Error: stale file" in output for output in tool_outputs)
    assert not any(
        "required_action_feedback" in str(message.get("content") or "")
        for message in agent.messages
        if message.get("role") == "user"
    )


def test_truncated_mutation_arguments_are_reported_then_regenerated(monkeypatch):
    monkeypatch.setenv("MYCODER_INJECTION_GUARD", "off")

    class _Edit(Tool):
        name = "edit_file"
        description = "edit"
        parameters = {"type": "object", "properties": {}}

        def __init__(self):
            self.calls = 0

        def execute(self):
            self.calls += 1
            return "Edited production.py"

    class _LLM:
        max_tokens = 1024

        def __init__(self):
            self.calls = 0

        def chat(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(
                    tool_calls=[
                        ToolCall(
                            id="truncated",
                            name="edit_file",
                            arguments={},
                            parse_error="tool arguments were truncated",
                        )
                    ]
                )
            if self.calls == 2:
                return LLMResponse(
                    tool_calls=[ToolCall(id="complete", name="edit_file", arguments={})]
                )
            return LLMResponse(content="implemented")

    edit = _Edit()
    agent = Agent(llm=_LLM(), tools=[edit], require_mutation=True)

    assert agent.chat("fix") == "implemented"
    assert edit.calls == 1
    assert [event["status"] for event in agent.tool_events] == ["error", "success"]
    outputs = [message["content"] for message in agent.messages if message.get("role") == "tool"]
    assert "INVALID_TOOL_INPUT" in outputs[0]


def test_completed_required_mutation_switches_to_tool_free_finalization(monkeypatch):
    monkeypatch.setenv("MYCODER_INJECTION_GUARD", "off")

    class _Edit(Tool):
        name = "edit_file"
        description = "edit"
        parameters = {"type": "object", "properties": {}}

        def execute(self):
            return "Edited app.py"

    class _LLM:
        max_tokens = 1024

        def __init__(self):
            self.requests = []

        def chat(self, **kwargs):
            self.requests.append(kwargs)
            if len(self.requests) == 1:
                return LLMResponse(
                    tool_calls=[ToolCall(id="edit", name="edit_file", arguments={})]
                )
            assert kwargs["tools"] == []
            return LLMResponse(content="implemented")

    llm = _LLM()
    agent = Agent(llm=llm, tools=[_Edit()], require_mutation=True)

    assert agent.chat("fix") == "implemented"
    assert len(llm.requests) == 2
    assert agent._last_convergence["stop_reason"] == "required postconditions completed"


def test_unmet_postcondition_never_gets_a_tool_free_summary_turn(monkeypatch):
    """P0-3: with tools=[] the model's DSML call is parsed as nothing.

    Spending the summary turn while a postcondition is still open is how a run
    that HAD asked for `execute_in_sandbox` ended up reported as "completed
    without a successful verification command". When the requirement is unmet
    the turn must end with a deterministic failure instead.
    """
    monkeypatch.setenv("MYCODER_INJECTION_GUARD", "off")

    class _Read(Tool):
        name = "read_file"
        description = "read"
        parameters = {"type": "object", "properties": {}}

        def execute(self):
            return "VALUE = 1"

    class _LLM:
        max_tokens = 1024

        def __init__(self):
            self.requests = []

        def chat(self, **kwargs):
            self.requests.append(kwargs)
            assert kwargs["tools"], "a tool-free request must not be made"
            return LLMResponse(
                tool_calls=[ToolCall(id=f"read-{len(self.requests)}", name="read_file", arguments={})]
            )

    llm = _LLM()
    agent = Agent(llm=llm, tools=[_Read()], require_mutation=True, max_rounds=2)

    result = agent.chat("fix")
    assert result.startswith("(required repository mutation not completed")
    assert "round limit reached" in result
    assert all(request["tools"] for request in llm.requests)


def test_tool_catalog_is_identical_on_every_round(monkeypatch):
    """P0-2: the model sees the same capabilities from the first round to the
    last, so the run does not depend on which round it happens to be in."""
    monkeypatch.setenv("MYCODER_INJECTION_GUARD", "off")

    class _Read(Tool):
        name = "read_file"
        description = "read"
        parameters = {"type": "object", "properties": {}}

        def execute(self):
            return "VALUE = 1"

    class _Edit(Tool):
        name = "edit_file"
        description = "edit"
        parameters = {"type": "object", "properties": {}}

        def execute(self):
            return "Edited app.py"

    class _Verify(Tool):
        name = "execute_in_sandbox"
        description = "verify"
        parameters = {"type": "object", "properties": {}}

        def execute(self, command):
            return ToolResult("ok", exit_code=0)

    class _LLM:
        max_tokens = 1024

        def __init__(self):
            self.catalogs = []

        def chat(self, **kwargs):
            self.catalogs.append(
                {schema["function"]["name"] for schema in kwargs["tools"]}
            )
            call = len(self.catalogs)
            if call == 1:
                return LLMResponse(tool_calls=[ToolCall(id="read", name="read_file", arguments={})])
            if call == 2:
                return LLMResponse(tool_calls=[ToolCall(id="edit", name="edit_file", arguments={})])
            if call == 3:
                return LLMResponse(
                    tool_calls=[
                        ToolCall(
                            id="verify",
                            name="execute_in_sandbox",
                            arguments={"command": "python -m pytest -q"},
                        )
                    ]
                )
            assert kwargs["tools"] == []
            self.catalogs[-1] = "<finalize>"
            return LLMResponse(content="done")

    llm = _LLM()
    agent = Agent(
        llm=llm,
        tools=[_Read(), _Edit(), _Verify()],
        require_mutation=True,
        require_verification=True,
    )

    assert agent.chat("fix") == "done"
    # Rounds 1-3 (explore / mutate / verify) advertise the identical catalog;
    # only the deliberate tool-free summary narrows it.
    assert llm.catalogs[:3] == [
        {"read_file", "edit_file", "execute_in_sandbox"},
        {"read_file", "edit_file", "execute_in_sandbox"},
        {"read_file", "edit_file", "execute_in_sandbox"},
    ]
    assert llm.catalogs[3] == "<finalize>"


def test_required_mutation_keeps_catalog_stable_across_stop_retry(monkeypatch):
    monkeypatch.setenv("MYCODER_INJECTION_GUARD", "off")

    class _Read(Tool):
        name = "read_file"
        description = "read"
        parameters = {"type": "object", "properties": {}}

        def __init__(self):
            self.calls = 0

        def execute(self):
            self.calls += 1
            return "VALUE = 1"

    class _Edit(Tool):
        name = "edit_file"
        description = "edit"
        parameters = {"type": "object", "properties": {}}

        def execute(self):
            return "Edited app.py"

    class _Verify(Tool):
        name = "execute_in_sandbox"
        description = "verify"
        parameters = {"type": "object", "properties": {}}

        def __init__(self):
            self.calls = 0

        def execute(self):
            self.calls += 1
            return "ok"

    class _LLM:
        def __init__(self):
            self.calls = 0

        def chat(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(content="done")
            if self.calls == 2:
                assert {schema["function"]["name"] for schema in kwargs["tools"]} == {
                    "read_file",
                    "edit_file",
                    "execute_in_sandbox",
                }
                return LLMResponse(tool_calls=[ToolCall(id="read-1", name="read_file", arguments={})])
            if self.calls == 3:
                return LLMResponse(
                    tool_calls=[ToolCall(id="verify", name="execute_in_sandbox", arguments={})]
                )
            if self.calls == 4:
                return LLMResponse(tool_calls=[ToolCall(id="edit", name="edit_file", arguments={})])
            return LLMResponse(content="implemented")

    read = _Read()
    verify = _Verify()
    agent = Agent(llm=_LLM(), tools=[read, _Edit(), verify], require_mutation=True)

    assert agent.chat("fix") == "implemented"
    assert read.calls == 1
    assert verify.calls == 1
    assert [event["status"] for event in agent.tool_events] == [
        "success",
        "success",
        "success",
    ]


def test_required_mutation_only_counts_real_file_mutations():
    """Only a mutation tool's success counts; nothing else can fake a diff."""
    assert Agent._is_required_mutation("write_file", "Wrote 3 lines") is True
    assert Agent._is_required_mutation("edit_file", "Edited app.py") is True
    assert Agent._is_required_mutation("read_file", "VALUE = 1") is False
    assert Agent._is_required_mutation("edit_file", "Error: not found") is False


def test_soft_budget_extension_preserves_evidence_tools(monkeypatch):
    monkeypatch.setenv("MYCODER_INJECTION_GUARD", "off")

    class _Guard:
        max_tokens_per_session = 20_000

        def __init__(self):
            self.remaining = 20_000

        def get_remaining(self, _session_id):
            return self.remaining

    class _WriteTool(Tool):
        name = "write_file"
        description = "write"
        parameters = {"type": "object", "properties": {}}

        def execute(self):
            return "Wrote 1 line"

    class _LLM:
        max_tokens = 256

        def __init__(self, guard):
            self.guard = guard
            self.calls = 0

        def chat(self, *, tools, **_kwargs):
            self.calls += 1
            names = {item["function"]["name"] for item in tools}
            if self.calls == 1:
                assert names == {"read", "write_file"}
                self.guard.remaining = 8_000
                return LLMResponse(tool_calls=[ToolCall(id="read", name="read", arguments={})])
            if self.calls == 2:
                # The first round already inspected evidence, so the required
                # action window keeps inspection available for safe edits.
                assert names == {"read", "write_file"}
                return LLMResponse(tool_calls=[ToolCall(id="write", name="write_file", arguments={})])
            assert tools == []
            return LLMResponse(content="implemented")

    from structlog.contextvars import bound_contextvars

    guard = _Guard()
    llm = _LLM(guard)
    agent = Agent(
        llm=llm,
        tools=[_ReadTool(), _WriteTool()],
        budget_guard=guard,
        require_mutation=True,
        max_turn_tokens=20_000,
        convergence_limits=ConvergenceLimits(soft_budget_ratio=0.5),
    )
    with bound_contextvars(session_id="action-window"):
        result = agent.chat("fix")

    assert result == "implemented"
    assert llm.calls == 3


def test_required_mutation_extension_is_bounded(monkeypatch):
    monkeypatch.setenv("MYCODER_INJECTION_GUARD", "off")

    class _Guard:
        max_tokens_per_session = 20_000

        def __init__(self):
            self.remaining = 20_000

        def get_remaining(self, _session_id):
            return self.remaining

    class _AnalysisOnlyLLM:
        max_tokens = 256

        def __init__(self, guard):
            self.calls = 0
            self.guard = guard

        def chat(self, **_kwargs):
            self.calls += 1
            self.guard.remaining = 8_000
            return LLMResponse(content="analysis only")

    from structlog.contextvars import bound_contextvars

    guard = _Guard()
    llm = _AnalysisOnlyLLM(guard)
    agent = Agent(
        llm=llm,
        tools=[],
        budget_guard=guard,
        require_mutation=True,
        max_turn_tokens=20_000,
        max_mutation_extension_rounds=2,
        max_mutation_feedback_rounds=10,
        convergence_limits=ConvergenceLimits(
            max_stagnant_rounds=10,
            soft_budget_ratio=0.5,
        ),
    )
    with bound_contextvars(session_id="bounded-action-window"):
        result = agent.chat("fix")

    assert "evidence-preserving extension exhausted" in result
    assert llm.calls == 3


def test_verification_requirement_accepts_executed_failing_test_as_evidence(monkeypatch):
    monkeypatch.setenv("MYCODER_INJECTION_GUARD", "off")

    class _CheckTool(Tool):
        name = "execute_in_sandbox"
        description = "check"
        parameters = {"type": "object", "properties": {}}
        idempotent = False

        def execute(self, command):
            # The real tool returns a ToolResult carrying the exit code; the
            # rendered "[exit code: N]" keeps the plain-string path readable.
            return ToolResult("1 failed\n[exit code: 1]", status="success", exit_code=1)

    class _LLM:
        def __init__(self):
            self.calls = 0

        def chat(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(
                    tool_calls=[
                        ToolCall(
                            id="test",
                            name="execute_in_sandbox",
                            arguments={"command": "pytest -q tests/test_bug.py"},
                        )
                    ]
                )
            return LLMResponse(content="verification failed")

    agent = Agent(
        llm=_LLM(),
        tools=[_CheckTool()],
        require_verification=True,
    )

    assert agent.chat("verify") == "verification failed"
    assert agent.verification_evidence[-1]["succeeded"] is False


def test_check_verdict_comes_from_exit_code_not_the_command_string():
    """P0-4: `./run_tests.sh` / `make check` are checks; `ls`/`grep` are not."""
    from mycoder.tools.base import ToolResult

    def evidence(command: str, exit_code: int) -> bool:
        return Agent._is_verification_evidence(
            "execute_in_sandbox",
            {"command": command},
            ToolResult("output", exit_code=exit_code),
        )

    # Any command that ran and reported an exit code is a check — including
    # the project-local runners the old allow-list could not see.
    assert evidence("./run_tests.sh", 0)
    assert evidence("make check", 0)
    assert evidence("python -m pytest -q tests/test_bug.py", 0)
    assert evidence("python -c \"from p import e; assert e('1') == '.----'\"", 0)
    # A red check is still a check that ran (the caller decides pass/fail).
    assert evidence("./run_tests.sh", 1)
    # Read-only / install commands say nothing about correctness.
    assert not evidence("ls -la", 0)
    assert not evidence("cat app.py", 0)
    assert not evidence("pip install -r requirements.txt", 0)
    assert not evidence("git status", 0)
    assert not evidence("cd /workspace && grep -n 'x' testing/test_y.py", 0)
    # A blocked or timed-out command never ran.
    assert not Agent._is_verification_evidence(
        "execute_in_sandbox",
        {"command": "pytest -q"},
        ToolResult("pytest -q\n[timed out]", status="success", exit_code=-1),
    )
    assert not Agent._is_verification_evidence(
        "execute_in_sandbox",
        {"command": "pytest -q"},
        ToolResult("⚠ Blocked: no", status="error"),
    )


def test_python_assert_script_is_verification_evidence():
    from mycoder.tools.base import ToolResult

    assert Agent._is_verification_evidence(
        "execute_in_sandbox",
        {"command": ("python -c \"from package import encode; assert encode('1') == '.----'\"")},
        ToolResult("(no output)", exit_code=0),
    )


def test_no_op_commands_are_not_verification_evidence():
    """A command that cannot fail cannot prove anything.

    Without this, `true` satisfies require_verification and MyCoder reports a
    success it never earned — the exact noise the exit-code rule was meant to
    remove. The exit code is 0 for all of these, so only the command shape
    distinguishes them from a real check.
    """
    from mycoder.tools.base import ToolResult

    def evidence(command: str) -> bool:
        return Agent._is_verification_evidence(
            "execute_in_sandbox",
            {"command": command},
            ToolResult("", exit_code=0),
        )

    for command in (
        "true",
        ":",
        "false",
        "sleep 0",
        "sleep 1",
        "exit",
        "exit 0",
        "bash -c true",
        'bash -c "true"',
        "sh -c ':'",
        "python -c pass",
        'python3 -c "pass"',
        "python -c ''",
        "sudo true",
        "cd /workspace && true",
        "env FOO=1 bash -c \"true\"",
    ):
        assert not evidence(command), command

    # A shell wrapper around a real command is still a real command.
    assert evidence('bash -c "pytest -q"')
    assert evidence("cd /workspace && ./run_tests.sh")
    # An inline python snippet that asserts something is a real check.
    assert evidence("python -c \"assert 1 == 1\"")


def test_read_only_grep_is_not_verification_evidence():
    """Mentioning a test path in grep must not satisfy the check contract."""
    from mycoder.tools.base import ToolResult

    assert not Agent._is_verification_evidence(
        "execute_in_sandbox",
        {"command": "cd /workspace && grep -n 'EncodedFile' testing/test_capture.py"},
        ToolResult("425:class EncodedFile", exit_code=0),
    )


def test_force_edit_allows_targeted_read_after_file_not_read_failure():
    assert Agent._mutation_needs_read_recovery(
        "Error [FILE_NOT_READ]: target must be read before it can be edited"
    )
    assert not Agent._mutation_needs_read_recovery("Error: invalid replacement")


def test_reserved_budget_prevents_implementer_from_consuming_verifier_share(monkeypatch):
    monkeypatch.setenv("MYCODER_INJECTION_GUARD", "off")

    class _Guard:
        max_tokens_per_session = 35_000

        @staticmethod
        def get_remaining(_session_id):
            return 9_000

    class _MustNotRun:
        max_tokens = 256

        def chat(self, **_kwargs):
            raise AssertionError("reserved verifier budget must not be consumed")

    from structlog.contextvars import bound_contextvars

    agent = Agent(
        llm=_MustNotRun(),
        tools=[],
        budget_guard=_Guard(),
        require_mutation=True,
        max_turn_tokens=20_000,
        reserved_tokens=10_000,
    )
    with bound_contextvars(session_id="reserved-budget"):
        result = agent.chat("fix")

    assert "reserved budget reached" in result


def test_requirement_check_does_not_double_count_current_verifier_reserve(monkeypatch):
    """A verifier still gets its first check when only its share remains."""
    monkeypatch.setenv("MYCODER_INJECTION_GUARD", "off")

    class _Guard:
        max_tokens_per_session = 35_000

        @staticmethod
        def get_remaining(_session_id):
            return 7_000

    from structlog.contextvars import bound_contextvars

    agent = Agent(
        llm=object(),
        tools=[],
        budget_guard=_Guard(),
        require_verification=True,
        verification_reserved_tokens=4_096,
    )
    with bound_contextvars(session_id="verifier-first-check"):
        assert agent._can_afford_requirement_round()


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


def test_new_agent_turn_does_not_inherit_prior_soft_budget_ratio(monkeypatch):
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
    assert agent._last_convergence["stop_reason"] == "model_completed"
