"""Trace replay: a recorded run is re-executed offline and compared.

The recordings here are produced by the *real* agent loop driven by a scripted
provider stand-in, so the tests exercise the production wiring (RecordingLLM +
Agent.run_recorder + RunLog) rather than a hand-written log fixture.
"""

import json

import pytest

from mycoder.agent import Agent
from mycoder.llm import LLMResponse, ToolCall
from mycoder.observability.budget import TokenBudgetGuard
from mycoder.observability.run_log import RecordingLLM, RunLog, RunLogRecorder
from mycoder.replay import (
    ReplayDivergence,
    ReplayLLM,
    STATUS_DIVERGED,
    STATUS_INCOMPLETE,
    STATUS_MATCH,
    STATUS_UNSUPPORTED,
    _reconstruct_messages,
    _Tracker,
    main,
    replay,
)
from mycoder.tools.file_state import FileStateTracker
from mycoder.tools.list_files import ListFilesTool
from mycoder.tools.read_file import ReadFileTool
from mycoder.tools.write import WriteFileTool

PROMPT = "What is in greeting.py?"


class _ScriptedLLM:
    """Provider stand-in: returns canned responses, never touches the network."""

    def __init__(self, responses, *, model="scripted-model", provider="scripted"):
        self.model = model
        self.provider = provider
        self.extra = {}
        self.tool_protocol = None
        self.requests = []
        self._responses = list(responses)

    def chat(
        self,
        messages,
        tools=None,
        on_token=None,
        response_format=None,
        predictive_executor=None,
        tool_choice=None,
        strict_tool_choice=False,
        timeout_seconds=None,
        request_max_retries=None,
    ):
        self.requests.append({"messages": list(messages), "tools": tools})
        if not self._responses:
            raise AssertionError("scripted provider ran out of responses")
        return self._responses.pop(0)


def _read_then_answer():
    return [
        LLMResponse(
            tool_calls=[ToolCall(id="call-1", name="read_file", arguments={"file_path": "greeting.py"})],
            prompt_tokens=120,
            completion_tokens=18,
        ),
        LLMResponse(content="greeting.py defines VALUE = 1.", prompt_tokens=200, completion_tokens=12),
    ]


def _tools(root):
    return [
        ReadFileTool(project_root=root, file_state=FileStateTracker()),
        WriteFileTool(project_root=root),
        ListFilesTool(project_root=root),
    ]


def _workspace(tmp_path):
    # Idempotent on purpose: some tests re-open the workspace `_record` already
    # made. Guarded with exists() rather than mkdir(exist_ok=True) so no second
    # mkdir syscall is issued at all.
    root = tmp_path / "repo"
    if not root.exists():
        root.mkdir(parents=True)
    (root / "greeting.py").write_text("VALUE = 1\n", encoding="utf-8")
    return root


def _record(tmp_path, responses, *, budget_guard=None, prompt=PROMPT, seen=None):
    """Run the real agent loop with recording on, exactly like the CLI does."""
    root = _workspace(tmp_path)
    log_path = tmp_path / "events.jsonl"
    recorder = RunLogRecorder(log_path, run_id="case-1", run_context={"execution_mode": "single"})
    inner = _ScriptedLLM(responses)
    agent = Agent(
        llm=RecordingLLM(inner, recorder),
        tools=_tools(root),
        max_rounds=4,
        budget_guard=budget_guard,
        run_recorder=recorder,
    )
    try:
        answer = agent.chat(prompt)
    finally:
        recorder.close()
    if seen is not None:
        seen["llm"] = inner
    return root, log_path, answer


# ------------------------------------------------------------------ replay


def test_replay_reproduces_a_recorded_run_exactly(tmp_path):
    root, log_path, answer = _record(tmp_path, _read_then_answer())

    report = replay(log_path, workspace=root, tools=_tools(root))

    assert report.status == STATUS_MATCH, report.render_markdown()
    assert report.matched and report.differences == []
    assert report.exchanges_served == report.exchanges_recorded == 2
    assert report.tool_calls_recorded == report.tool_calls_replayed == 1
    assert report.tool_calls_matched == 1
    assert report.answer == answer == "greeting.py defines VALUE = 1."
    assert report.replay_answer == answer
    assert report.recorded_status == "ok"
    # The replay writes its own log next to the recording, for direct diffing.
    assert report.output_log and report.output_log.endswith(".replay.jsonl")
    # A run that actually executed may claim the action layer was verified.
    assert report.executed is True
    assert "# Trace replay — MATCH" in report.render_markdown()
    assert "reproduced the recording exactly" in report.render_markdown()


def test_replay_reports_the_run_start_contract_it_rebuilt(tmp_path):
    root, log_path, _ = _record(tmp_path, _read_then_answer())
    log = RunLog.load(log_path)
    flags = log.flags

    assert flags["execution_mode"] == "single"
    assert flags["require_mutation"] is False
    assert flags["require_verification"] is False
    assert flags["max_rounds"] == 4
    assert flags["max_context_tokens"] > 0
    assert flags["soft_budget_ratio"] is not None
    assert flags["memory"] is False
    assert log.tool_names and "read_file" in log.tool_names
    assert log.multi_agent is False
    assert [item.name for item in log.tool_results] == ["read_file"]


def test_replay_pins_the_sandbox_policy_from_the_api_side(tmp_path):
    root = _workspace(tmp_path)
    log_path = tmp_path / "events.jsonl"
    recorder = RunLogRecorder(
        log_path,
        run_context={"execution_mode": "single", "sandbox_policy": "interactive"},
    )
    agent = Agent(
        llm=RecordingLLM(_ScriptedLLM(_read_then_answer()), recorder),
        tools=_tools(root),
        max_rounds=4,
        run_recorder=recorder,
    )
    agent.chat(PROMPT)
    recorder.close()

    report = replay(log_path, workspace=root, tools=_tools(root), sandbox_policy="benchmark")
    # The recording says "interactive"; running it under another policy is
    # reported instead of silently producing a different answer.
    assert any("sandbox policy differs" in note for note in report.notes)


def test_replay_detects_a_changed_tool_result_and_pinpoints_it(tmp_path):
    root, log_path, _ = _record(tmp_path, _read_then_answer())
    (root / "greeting.py").write_text("VALUE = 2\n", encoding="utf-8")

    report = replay(log_path, workspace=root, tools=_tools(root))

    assert report.status == STATUS_DIVERGED
    first = report.first_difference
    assert first is not None and first.kind == "tool_result"
    assert "VALUE = 1" in first.expected and "VALUE = 2" in first.actual
    # The changed observation also changes the request the loop sends next.
    assert any(item.kind == "llm_request" for item in report.differences)
    assert report.tool_calls_matched == 0
    rendered = report.render_markdown()
    assert "DIVERGED" in rendered and "recorded:" in rendered and "replayed:" in rendered
    assert report.to_dict()["first_difference"]["kind"] == "tool_result"


def test_replay_reports_a_loop_that_runs_past_the_recording(tmp_path):
    _, log_path, _ = _record(tmp_path, _read_then_answer())
    lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    kept = [line for line in lines if json.loads(line)["type"] not in {"llm_call", "run_end"} or json.loads(line)["seq"] < 2]
    log_path.write_text("\n".join(kept) + "\n", encoding="utf-8")

    report = replay(log_path, tools=_tools(_workspace(tmp_path)))

    assert report.status == STATUS_DIVERGED
    assert any("past the recording" in item.summary for item in report.differences)
    assert report.exchanges_served == report.exchanges_recorded == 1


def test_replay_keeps_a_clean_truncated_prefix_incomplete(tmp_path):
    """A replayable prefix must not be presented as a complete verification."""
    root, log_path, _ = _record(tmp_path, [LLMResponse(content="done")])
    lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    # Keep the full exchange but remove run_end. The replay can reproduce the
    # prefix exactly, yet the original run's outcome is unknown.
    kept = [
        line
        for line in lines
        if json.loads(line)["type"] != "run_end"
    ]
    log_path.write_text("\n".join(kept) + "\n", encoding="utf-8")

    report = replay(log_path, tools=_tools(root))

    assert report.status == STATUS_INCOMPLETE, report.render_markdown()
    assert report.differences == []
    assert "no run_end event" in "\n".join(report.notes)


def test_replay_reproduces_a_recorded_provider_failure(tmp_path):
    class _Boom:
        def __init__(self):
            self.model = "scripted-model"
            self.provider = "scripted"
            self.extra = {}
            self.tool_protocol = None

        def chat(self, *_args, **_kwargs):
            raise TimeoutError("provider stalled")

    root = _workspace(tmp_path)
    log_path = tmp_path / "events.jsonl"
    recorder = RunLogRecorder(log_path, run_context={"execution_mode": "single"})
    agent = Agent(llm=RecordingLLM(_Boom(), recorder), tools=_tools(root), run_recorder=recorder)
    with pytest.raises(TimeoutError):
        agent.chat(PROMPT)
    recorder.close()

    report = replay(log_path, workspace=root, tools=_tools(root))

    assert report.status == STATUS_MATCH, report.render_markdown()
    assert report.recorded_status == "error"
    assert any("provider failure was reproduced" in note for note in report.notes)


def test_replay_refuses_a_delegated_run_instead_of_guessing(tmp_path):
    recorder = RunLogRecorder(tmp_path / "events.jsonl", run_context={"execution_mode": "multi"})
    recorder.record_run_start(prompt="p", model="m", tools=["read_file"])
    recorder.record_llm_call(messages=[{"role": "user", "content": "p"}], response=LLMResponse(content="hi"))
    recorder.record_run_end(status="ok", answer="hi")
    recorder.close()

    report = replay(tmp_path / "events.jsonl")

    assert report.status == STATUS_UNSUPPORTED
    assert report.mode == "multi"
    assert report.output_log is None
    assert any("not replayable yet" in note for note in report.notes)


def test_replay_can_analyse_without_executing(tmp_path):
    root, log_path, _ = _record(tmp_path, _read_then_answer())
    report = replay(log_path, workspace=root, tools=_tools(root), execute=False)

    assert report.status == STATUS_MATCH
    assert report.exchanges_served == 0
    assert report.output_log is None
    assert any("nothing was re-run" in note for note in report.notes)
    # Analysing is not verifying: the headline must not read as a match.
    rendered = report.render_markdown()
    assert report.executed is False
    assert "# Trace replay — MATCH" not in rendered
    assert "ANALYSED (not re-run)" in rendered
    assert "reproduced the recording exactly" not in rendered


def test_replay_selects_the_requested_turn(tmp_path):
    root = _workspace(tmp_path)
    log_path = tmp_path / "events.jsonl"
    recorder = RunLogRecorder(log_path, run_context={"execution_mode": "single"})
    first = _ScriptedLLM([LLMResponse(content="first answer")])
    second = _ScriptedLLM([LLMResponse(content="second answer")])
    for inner, prompt in ((first, "first question"), (second, "second question")):
        agent = Agent(llm=RecordingLLM(inner, recorder), tools=_tools(root), run_recorder=recorder)
        agent.chat(prompt)
    recorder.close()

    assert RunLog.load(log_path).run_count == 2
    report = replay(log_path, run_index=1, workspace=root, tools=_tools(root))
    assert report.status == STATUS_MATCH
    assert report.answer == "second answer"
    assert report.exchanges_recorded == 1


def test_replay_reconstructs_the_recorded_conversation(tmp_path):
    """Divergence messages quote recorded text, so it has to be recoverable."""
    root, log_path, _ = _record(tmp_path, _read_then_answer())
    log = RunLog.load(log_path)

    first = _reconstruct_messages(log, log.exchanges[0])
    second = _reconstruct_messages(log, log.exchanges[1])
    assert first is not None and second is not None
    from mycoder.observability.run_log import digest_messages

    assert digest_messages(second)[1:] == list(log.exchanges[1].digests[1:])
    # prompt -> assistant(tool call) -> tool observation -> assistant(answer)
    assert [message["role"] for message in second] == ["system", "user", "assistant", "tool"]
    assert "<tool_output" in second[3]["content"]
    assert second[2]["tool_calls"][0]["function"]["name"] == "read_file"


def test_replay_llm_feeds_recorded_usage_to_the_budget_guard(tmp_path):
    seen = {}
    root, log_path, _ = _record(
        tmp_path, _read_then_answer(), budget_guard=TokenBudgetGuard(max_tokens_per_session=100_000), seen=seen
    )
    log = RunLog.load(log_path)
    request = seen["llm"].requests[0]

    guard = TokenBudgetGuard(max_tokens_per_session=100_000)
    llm = ReplayLLM(log, budget_guard=guard, session_id="s-1")
    assert llm.remaining == 2
    llm.chat(messages=request["messages"], tools=request["tools"])
    # 120 + 18 from the first recorded exchange, replayed as the guard saw it.
    assert guard.get_used("s-1") == 138
    assert log.flags["budget_max_tokens"] == 100_000


def test_replay_llm_does_not_build_tier_clients(tmp_path):
    """A tier lookup during replay must not construct a second real client."""
    root, log_path, _ = _record(tmp_path, _read_then_answer())
    llm = ReplayLLM(RunLog.load(log_path))
    assert llm.builds_tier_clients is False

    from mycoder.model_router import build_model_factory

    factory = build_model_factory(llm, router=_Router("tier-model"))
    # None is the factory's way of saying "keep using the shared LLM" — which is
    # the replay, instead of a freshly built provider client.
    assert factory("fast") is None


class _Router:
    def __init__(self, model):
        self.model = model

    def resolve_candidates(self, tier, provider, base_model):
        return [self.model]


def test_replay_llm_rejects_an_unrecorded_request(tmp_path):
    seen = {}
    root, log_path, _ = _record(tmp_path, _read_then_answer(), seen=seen)
    log = RunLog.load(log_path)
    first, second = seen["llm"].requests[0], seen["llm"].requests[1]
    llm = ReplayLLM(log)
    llm.chat(messages=first["messages"], tools=first["tools"])

    from mycoder.replay import ReplayDivergence

    with pytest.raises(ReplayDivergence, match="does not match the recorded request"):
        llm.chat(messages=[{"role": "user", "content": "something else"}])
    with pytest.raises(ReplayDivergence, match="first differing message"):
        llm.chat(messages=[{"role": "user", "content": "something else"}, {"role": "user", "content": "x"}])
    assert llm.remaining == 1
    # The recorded exchange is still served after a rejected request: a
    # divergence is reported, it does not corrupt the replay's position.
    llm.chat(messages=second["messages"], tools=second["tools"])
    with pytest.raises(ReplayDivergence, match="past the recording"):
        llm.chat(messages=second["messages"], tools=second["tools"])


def test_replay_fingerprints_provider_request_options(tmp_path):
    path = tmp_path / "options.jsonl"
    recorder = RunLogRecorder(path, run_context={"execution_mode": "single"})
    messages = [{"role": "user", "content": "return json"}]
    recorder.record_run_start(prompt="return json", model="m")
    recorder.record_llm_call(
        messages=messages,
        response=LLMResponse(content='{"ok":true}'),
        response_format={"type": "json_object"},
        tool_choice={"type": "function", "function": {"name": "answer"}},
        strict_tool_choice=True,
    )
    recorder.record_run_end(status="ok", answer='{"ok":true}')
    recorder.close()
    log = RunLog.load(path)

    with pytest.raises(ReplayDivergence, match="response_format changed"):
        ReplayLLM(log).chat(messages=messages)

    with pytest.raises(ReplayDivergence, match="tool_choice changed"):
        ReplayLLM(log).chat(
            messages=messages,
            response_format={"type": "json_object"},
            tool_choice={"type": "function", "function": {"name": "other"}},
            strict_tool_choice=True,
        )

    response = ReplayLLM(log).chat(
        messages=messages,
        response_format={"type": "json_object"},
        tool_choice={"type": "function", "function": {"name": "answer"}},
        strict_tool_choice=True,
    )
    assert response.content == '{"ok":true}'


# --------------------------------------------------------- comparison units


def _synthetic_log(tmp_path, *, content="ok", status="success", mutation=True, name="read_file"):
    path = tmp_path / "events.jsonl"
    recorder = RunLogRecorder(path, run_context={"execution_mode": "single"})
    recorder.record_run_start(prompt="p", model="m")
    recorder.record_tool_result(
        name=name,
        tool_call_id="c1",
        arguments={"file_path": "a.py"},
        content=content,
        status=status,
        mutation=mutation,
    )
    recorder.record_run_end(status="ok", answer="done")
    recorder.close()
    return RunLog.load(path)


def _payload(**overrides):
    base = {
        "name": "read_file",
        "tool_call_id": "c1",
        "arguments": {"file_path": "a.py"},
        "content": "ok",
        "status": "success",
        "mutation": True,
        "verification": False,
        "blocked": False,
    }
    base.update(overrides)
    return base


def test_tracker_accepts_an_identical_tool_result(tmp_path):
    tracker = _Tracker(_synthetic_log(tmp_path))
    assert tracker.observe_tool_result(_payload()) == []
    assert tracker.tools_compared == 1


def test_tracker_reports_content_sequence_status_and_flags(tmp_path):
    log = _synthetic_log(tmp_path)

    changed = _Tracker(log).observe_tool_result(_payload(content="ok now different"))
    assert [item.kind for item in changed] == ["tool_result"]
    assert changed[0].index == 0

    sequence = _Tracker(log).observe_tool_result(_payload(name="write_file"))
    assert [item.kind for item in sequence] == ["tool_sequence"]

    status = _Tracker(log).observe_tool_result(_payload(status="error"))
    assert [item.kind for item in status] == ["tool_status"]

    flags = _Tracker(log).observe_tool_result(_payload(mutation=False))
    assert [item.kind for item in flags] == ["tool_flags"]


def test_tracker_reports_extra_and_missing_tool_calls(tmp_path):
    log = _synthetic_log(tmp_path)

    extra = _Tracker(log)
    extra.observe_tool_result(_payload())
    assert [item.kind for item in extra.observe_tool_result(_payload())] == ["extra_tool_call"]

    missing = _Tracker(log)
    trailing = missing.trailing()
    assert [item.kind for item in trailing] == ["missing_tool_call"]
    assert trailing[0].expected == "ok"


def test_tracker_reports_a_changed_control_message(tmp_path):
    path = tmp_path / "events.jsonl"
    recorder = RunLogRecorder(path, run_context={"execution_mode": "single"})
    recorder.record_run_start(prompt="p", model="m")
    recorder.record_control(kind="mutation_feedback", content="edit the file")
    recorder.record_control(kind="requirement_prompt", content="verify the change")
    recorder.record_run_end(status="ok", answer="done")
    recorder.close()

    tracker = _Tracker(RunLog.load(path))
    assert tracker.observe_control({"kind": "mutation_feedback", "content": "edit the file"}) == []
    changed = tracker.observe_control({"kind": "requirement_prompt", "content": "something else"})
    assert [item.kind for item in changed] == ["control"]
    assert changed[0].expected == "verify the change"
    assert [item.kind for item in tracker.observe_control({})] == ["extra_control"]


# ---------------------------------------------------------------------- CLI


def test_cli_lists_runs_and_checks_a_log_without_executing(tmp_path, capsys):
    root, log_path, _ = _record(tmp_path, _read_then_answer())

    assert main([str(log_path), "--list-runs"]) == 0
    assert "#0" in capsys.readouterr().out

    assert main([str(log_path), "--no-execute"]) == 0
    assert "nothing was re-run" in capsys.readouterr().out


def test_cli_exit_codes_describe_the_verdict(tmp_path, capsys):
    root, log_path, _ = _record(tmp_path, _read_then_answer())

    # A log with no run at all is an error (2), not a silent success.
    assert main([str(tmp_path / "missing.jsonl")]) == 2
    # A delegated recording cannot be replayed: also a non-zero verdict, and the
    # reason is printed rather than a fabricated match.
    assert main([str(log_path), "--mode", "multi", "--no-execute"]) == 2
    assert "not replayable yet" in capsys.readouterr().out

    # The exit code follows the status the report carries.
    from mycoder.replay import _EXIT_CODES, STATUS_ERROR

    assert _EXIT_CODES[STATUS_DIVERGED] == 1
    assert _EXIT_CODES[STATUS_MATCH] == 0
    assert _EXIT_CODES[STATUS_ERROR] == 2


def test_cli_writes_a_json_report(tmp_path):
    root, log_path, _ = _record(tmp_path, _read_then_answer())
    out = tmp_path / "report.json"

    assert main([str(log_path), "--workspace", str(root), "--json", str(out), "--no-execute"]) == 0

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["status"] == STATUS_MATCH
    assert payload["exchanges"]["recorded"] == 2
    assert payload["matched"] is True
    # A consumer must be able to tell an analysis from a verified replay.
    assert payload["executed"] is False
