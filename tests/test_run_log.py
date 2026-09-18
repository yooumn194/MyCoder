"""Run log (the recording half of trace replay): writing, reading, transparency."""

import json

import pytest

from mycoder.llm import LLMResponse, ToolCall
from mycoder.observability.run_log import (
    RecordingLLM,
    RunLog,
    RunLogError,
    RunLogRecorder,
    digest_messages,
    digest_tools,
    llm_core,
    llm_like,
    recording_model_factory,
)


class _StubLLM:
    """Minimal provider stand-in: canned responses, no network, no client."""

    def __init__(self, responses, *, model="stub-model", provider="stub"):
        self.model = model
        self.provider = provider
        self.extra = {"temperature": 0}
        self.tool_protocol = None
        self.calls = []
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
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "timeout_seconds": timeout_seconds,
                "request_max_retries": request_max_retries,
            }
        )
        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def _exchange(messages, *, content=""):
    return LLMResponse(
        content=content,
        tool_calls=[ToolCall(id="c1", name="read_file", arguments={"file_path": "a.py"})],
        prompt_tokens=11,
        completion_tokens=7,
    )


# --------------------------------------------------------------- recording


def test_recorder_appends_one_json_object_per_event(tmp_path):
    recorder = RunLogRecorder(tmp_path / "events.jsonl")
    recorder.record_run_start(prompt="hello", model="m", provider="p", tools=["read_file"])
    recorder.record_llm_call(messages=[{"role": "user", "content": "hello"}], response=_exchange([]))
    recorder.record_control(kind="mutation_feedback", content="edit something")
    recorder.record_run_end(status="ok", answer="done")
    recorder.close()

    lines = (tmp_path / "events.jsonl").read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 4
    events = [json.loads(line) for line in lines]
    assert [event["seq"] for event in events] == [0, 1, 2, 3]
    assert [event["type"] for event in events] == ["run_start", "llm_call", "control", "run_end"]


def test_recorder_digests_the_request_and_keeps_the_response(tmp_path):
    recorder = RunLogRecorder(tmp_path / "events.jsonl")
    recorder.record_run_start(prompt="hi", model="stub-model")
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    tools = [{"type": "function", "function": {"name": "read_file"}}]
    recorder.record_llm_call(messages=messages, tools=tools, response=_exchange(messages), caller="agent")
    recorder.close()

    log = RunLog.load(tmp_path / "events.jsonl")
    exchange = log.exchanges[0]
    # A fingerprint, not a copy: the conversation is compared, not stored twice.
    assert len(exchange.digests) == 2 and "sys" not in json.dumps(exchange.raw)
    assert exchange.digests == tuple(digest_messages(messages))
    assert exchange.tools_digest == digest_tools(tools)
    assert exchange.tools_count == 1
    assert exchange.caller == "agent"
    assert exchange.tool_calls[0]["name"] == "read_file"
    assert exchange.response["prompt_tokens"] == 11
    assert exchange.matches(messages, tools)


def test_recorder_keeps_a_provider_failure_and_its_digests(tmp_path):
    recorder = RunLogRecorder(tmp_path / "events.jsonl")
    recorder.record_run_start(prompt="hi", model="stub-model")
    recorder.record_llm_call(
        messages=[{"role": "user", "content": "hi"}],
        error="RateLimitError: slow down",
        duration_ms=12.5,
    )
    recorder.close()

    exchange = RunLog.load(tmp_path / "events.jsonl").exchanges[0]
    assert exchange.response is None
    assert exchange.error == "RateLimitError: slow down"
    assert exchange.message_count == 1


def test_recorder_survives_an_unwritable_target(tmp_path):
    """Recording is observation-only: it must never break the run it watches."""
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory", encoding="utf-8")
    recorder = RunLogRecorder(blocker / "events.jsonl")
    assert recorder.write({"type": "run_start", "schema": 1}) is None
    assert recorder.write({"type": "run_end"}) is None  # stays disabled, no raise


def test_recorder_run_context_reaches_the_flags(tmp_path):
    recorder = RunLogRecorder(
        tmp_path / "events.jsonl",
        run_id="swe-1",
        run_context={"execution_mode": "multi", "sandbox_policy": "benchmark"},
    )
    recorder.record_run_start(prompt="p", model="m", flags={"require_mutation": True})
    recorder.close()

    log = RunLog.load(tmp_path / "events.jsonl")
    assert log.flags["execution_mode"] == "multi"
    assert log.flags["sandbox_policy"] == "benchmark"
    assert log.flags["require_mutation"] is True
    assert log.multi_agent is True


def test_recorder_names_the_file_after_the_run(tmp_path):
    recorder = RunLogRecorder(tmp_path, run_id="swe-django__django-11133")
    assert recorder.path.name == "swe-django__django-11133.jsonl"
    recorder.close()


# ----------------------------------------------------------------- reading


def test_run_log_loads_windows_and_reports_completeness(tmp_path):
    path = tmp_path / "events.jsonl"
    recorder = RunLogRecorder(path)
    for prompt in ("first", "second"):
        recorder.record_run_start(prompt=prompt, model="m")
        recorder.record_run_end(status="ok", answer=f"answer to {prompt}")
    recorder.close()

    log = RunLog.load(path)
    assert log.run_count == 2
    assert log.prompt == "first"
    assert log.complete
    second = log.window(1)
    assert second.prompt == "second"
    assert second.answer == "answer to second"


def test_nested_end_does_not_close_the_parent_run(tmp_path):
    path = tmp_path / "events.jsonl"
    recorder = RunLogRecorder(path)
    recorder.record_run_start(prompt="parent", model="m", nested=False)
    recorder.record_run_start(prompt="child", model="m", nested=True)
    recorder.record_run_end(status="ok", answer="child", nested=True)
    recorder.close()

    log = RunLog.load(path)
    assert log.run_end is None
    assert log.complete is False
    assert len(log.nested_starts) == 1
    assert log.multi_agent is True


def test_run_log_accepts_a_torn_tail_but_not_a_torn_middle(tmp_path):
    path = tmp_path / "events.jsonl"
    recorder = RunLogRecorder(path)
    recorder.record_run_start(prompt="p", model="m")
    recorder.record_llm_call(messages=[{"role": "user", "content": "p"}], response=_exchange([]))
    recorder.close()
    body = path.read_text(encoding="utf-8")

    path.write_text(body + '{"type": "llm_call", "seq": 2, "req', encoding="utf-8")
    partial = RunLog.load(path)
    assert partial.partial_tail
    assert len(partial.exchanges) == 1  # the prefix stays replayable

    path.write_text(body.replace("\n", "\nnot json\n", 1), encoding="utf-8")
    with pytest.raises(RunLogError, match="malformed run log line"):
        RunLog.load(path)


def test_torn_tail_only_marks_the_unfinished_window_incomplete(tmp_path):
    path = tmp_path / "events.jsonl"
    recorder = RunLogRecorder(path)
    recorder.record_run_start(prompt="first", model="m")
    recorder.record_run_end(status="ok", answer="done")
    recorder.record_run_start(prompt="second", model="m")
    recorder.close()
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"type":"llm_call","seq":3,"request":')

    log = RunLog.load(path)
    assert log.window(0).complete is True
    assert log.window(1).complete is False


def test_run_log_rejects_unknown_and_missing_schemas(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps({"type": "run_start", "schema": 99, "prompt": "p"}) + "\n", encoding="utf-8")
    with pytest.raises(RunLogError, match="newer than"):
        RunLog.load(path)

    path.write_text(json.dumps({"type": "llm_call", "seq": 0}) + "\n", encoding="utf-8")
    with pytest.raises(RunLogError, match="no run_start"):
        RunLog.load(path)

    with pytest.raises(RunLogError, match="not found"):
        RunLog.load(tmp_path / "absent.jsonl")


def test_exchange_locates_the_first_differing_message(tmp_path):
    recorder = RunLogRecorder(tmp_path / "events.jsonl")
    recorder.record_run_start(prompt="keep", model="stub-model")
    messages = [{"role": "user", "content": "keep"}, {"role": "tool", "content": "original"}]
    recorder.record_llm_call(messages=messages, response=_exchange(messages))
    recorder.close()

    exchange = RunLog.load(tmp_path / "events.jsonl").exchanges[0]
    changed = [{"role": "user", "content": "keep"}, {"role": "tool", "content": "different"}]
    assert exchange.locate(changed) == 1
    assert exchange.locate(messages) is None
    assert exchange.locate(messages + [{"role": "tool", "content": "extra"}]) == 2
    assert not exchange.matches(changed, None)


# ------------------------------------------------------- RecordingLLM


def test_recording_llm_forwards_every_call_option_and_records_once(tmp_path):
    inner = _StubLLM([_exchange([], content="ok")])
    recorder = RunLogRecorder(tmp_path / "events.jsonl")
    recorder.record_run_start(prompt="hi", model="stub-model")
    llm = RecordingLLM(inner, recorder)

    response = llm.chat(
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        timeout_seconds=30.0,
        request_max_retries=1,
    )

    assert response.content == "ok"
    assert inner.calls[0]["timeout_seconds"] == 30.0
    assert inner.calls[0]["request_max_retries"] == 1
    log = RunLog.load(tmp_path / "events.jsonl")
    assert len(log.exchanges) == 1 and log.exchanges[0].message_count == 1


def test_recording_llm_delegates_attributes_and_forwards_model_writes(tmp_path):
    inner = _StubLLM([], model="base-model")
    llm = RecordingLLM(inner, None)

    assert llm.model == "base-model"
    assert llm.provider == "stub"
    assert llm.extra == {"temperature": 0}
    assert llm.tool_protocol is None

    llm.model = "switched-model"
    assert inner.model == "switched-model", "a delegated write must reach the provider client"
    assert llm.model == "switched-model"


def test_recording_llm_has_no_recorder_path_when_disabled(tmp_path):
    inner = _StubLLM([_exchange([], content="ok")])
    llm = RecordingLLM(inner, None)
    assert llm.chat(messages=[{"role": "user", "content": "hi"}]).content == "ok"
    assert llm.run_log_path is None
    assert not (tmp_path / "events.jsonl").exists()


def test_recording_llm_records_the_failure_then_reraises(tmp_path):
    inner = _StubLLM([TimeoutError("provider stalled")])
    recorder = RunLogRecorder(tmp_path / "events.jsonl")
    recorder.record_run_start(prompt="hi", model="stub-model")
    llm = RecordingLLM(inner, recorder)

    with pytest.raises(TimeoutError, match="provider stalled"):
        llm.chat(messages=[{"role": "user", "content": "hi"}])

    exchange = RunLog.load(tmp_path / "events.jsonl").exchanges[0]
    assert exchange.error == "TimeoutError: provider stalled"
    assert exchange.response is None


def test_recording_llm_exposes_the_core_and_the_recorder(tmp_path):
    inner = _StubLLM([])
    recorder = RunLogRecorder(tmp_path / "events.jsonl")
    llm = RecordingLLM(inner, recorder)

    assert llm_core(llm) is inner
    assert llm.run_log_recorder is recorder
    assert llm_core(inner) is inner


def test_llm_like_restores_the_decorator_chain(tmp_path):
    recorder = RunLogRecorder(tmp_path / "events.jsonl")
    template = RecordingLLM(_StubLLM([]), recorder)
    replacement = _StubLLM([], model="tier-model")

    rewrapped = llm_like(template, replacement)
    assert isinstance(rewrapped, RecordingLLM)
    assert llm_core(rewrapped) is replacement
    # Already wrapped stays as it is (no double recording).
    assert llm_like(template, rewrapped) is rewrapped
    # A plain template leaves the new client alone.
    assert llm_like(_StubLLM([]), replacement) is replacement


def test_recording_model_factory_wraps_only_fresh_clients(tmp_path):
    recorder = RunLogRecorder(tmp_path / "events.jsonl")
    tier_client = _StubLLM([], model="fast-model")

    factory = recording_model_factory(lambda tier: tier_client if tier == "fast" else None, recorder)
    wrapped = factory("fast")
    assert isinstance(wrapped, RecordingLLM) and llm_core(wrapped) is tier_client
    assert factory("unknown") is None
    assert recording_model_factory(None, recorder) is None
