"""Tests for core modules: config, context, session, imports."""

import pytest

from mycoder import Agent, LLM, Config, ALL_TOOLS, __version__
from mycoder import session as session_module
from mycoder.context import ContextManager, estimate_tokens
from mycoder.llm import LLMResponse, ToolCall
from mycoder.session import save_session, load_session, list_sessions
from mycoder.tools import get_tool


def test_version():
    assert __version__ == "0.5.0"


def test_reasoning_only_response_keeps_valid_assistant_history_message():
    response = LLMResponse(reasoning_content="private reasoning")

    assert response.message == {
        "role": "assistant",
        "content": "No visible response; continue.",
        "reasoning_content": "private reasoning",
    }


def test_tool_call_response_may_keep_null_content():
    response = LLMResponse(tool_calls=[ToolCall(id="call-1", name="read_file", arguments={})])

    assert response.message["content"] is None
    assert response.message["tool_calls"][0]["id"] == "call-1"


def test_public_api_exports():
    """Users should be able to import key classes from the top-level package."""
    assert Agent is not None
    assert LLM is not None
    assert Config is not None
    # bash was replaced by execute_in_sandbox; grep_search, list_files,
    # fetch_url, and the Phase 3 planning tools added
    assert {t.name for t in ALL_TOOLS} == {
        "execute_in_sandbox",
        "grep_search",
        "list_files",
        "read_file",
        "write_file",
        "edit_file",
        "glob",
        "grep",
        "agent",
        "spawn_subagent",
        "fetch_url",
        "todo_write",
        "todo_update",
        # Phase 5 memory tools
        "memory_save",
        "memory_search",
        "memory_list",
        "memory_forget",
        "memory_confirm",
        "memory_correct",
        "memory_stats",
    }


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("MYCODER_MODEL", "test-model")
    c = Config.from_env()
    assert c.model == "test-model"


def test_config_defaults(monkeypatch):
    # clear relevant env vars without leaking the change into other tests
    monkeypatch.delenv("MYCODER_MODEL", raising=False)
    monkeypatch.delenv("MYCODER_MAX_TOKENS", raising=False)

    c = Config.from_env()
    assert c.model == "gpt-5.5"
    assert c.max_tokens == 4096
    assert c.temperature == 0.0


def test_deepseek_thinking_mode_is_provider_aware(monkeypatch):
    monkeypatch.setenv("MYCODER_PROFILE", "deepseek")
    monkeypatch.setenv("MYCODER_DEEPSEEK_THINKING", "disabled")
    c = Config.from_env()
    assert c.thinking == "disabled"
    assert c.tool_dialect == "zcode"


def test_openrouter_config_from_provider_env(monkeypatch):
    monkeypatch.setenv("MYCODER_PROVIDER", "OpenRouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")

    c = Config.from_env()

    assert c.provider == "openrouter"
    assert c.api_key == "or-key"
    assert c.base_url == "https://openrouter.ai/api/v1"
    assert c.model == "minimax/minimax-m3:free"
    assert c.tool_dialect == "auto"


def test_provider_profile_ignores_stale_generic_model_and_endpoint(monkeypatch):
    monkeypatch.setenv("MYCODER_PROFILE", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("MYCODER_API_KEY", "stale-generic-key")
    monkeypatch.setenv("MYCODER_MODEL", "deepseek-v4-flash")
    monkeypatch.setenv("MYCODER_BASE_URL", "https://api.deepseek.com")

    c = Config.from_env()

    assert c.provider == "openrouter"
    assert c.api_key == "or-key"
    assert c.model == "minimax/minimax-m3:free"
    assert c.base_url == "https://openrouter.ai/api/v1"


def test_provider_profile_supports_provider_specific_model_override(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("MYCODER_MODEL", "stale-model")
    monkeypatch.setenv("MYCODER_OPENROUTER_MODEL", "openai/gpt-oss-120b:free")

    c = Config.from_env(provider_override="openrouter")

    assert c.provider == "openrouter"
    assert c.model == "openai/gpt-oss-120b:free"


def test_unknown_provider_profile_fails_fast(monkeypatch):
    import pytest

    monkeypatch.setenv("MYCODER_PROFILE", "typo-provider")

    with pytest.raises(ValueError, match="unknown provider profile"):
        Config.from_env()


def test_auto_model_value_uses_detected_provider_default(monkeypatch):
    monkeypatch.setenv("MYCODER_PROVIDER", "deepseek")
    monkeypatch.setenv("MYCODER_MODEL", "auto")

    c = Config.from_env()

    assert c.model == "deepseek-flash"


def test_cli_accepts_provider_profile(monkeypatch):
    import sys

    from mycoder.cli import _parse_args

    monkeypatch.setattr(sys, "argv", ["mycoder", "--provider", "openrouter"])
    args = _parse_args()

    assert args.provider == "openrouter"


def test_openrouter_provider_is_auto_detected(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")

    c = Config.from_env()

    assert c.provider == "openrouter"


def test_deepseek_is_detected_from_compatible_base_url(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("OPENAI_API_KEY", "legacy-compatible-key")

    c = Config.from_env()

    assert c.provider == "deepseek"
    assert c.api_key == "legacy-compatible-key"
    assert c.model == "deepseek-flash"


def test_openrouter_optional_attribution_headers(monkeypatch):
    from unittest.mock import patch

    from mycoder.llm import LLM

    monkeypatch.setenv("OPENROUTER_SITE_URL", "https://example.test")
    monkeypatch.setenv("OPENROUTER_APP_NAME", "CoreCoder")
    with patch("mycoder.llm.OpenAI") as openai:
        LLM(model="m", api_key="k", provider="openrouter")

    assert openai.call_args.kwargs["default_headers"] == {
        "HTTP-Referer": "https://example.test",
        "X-OpenRouter-Title": "CoreCoder",
    }


def test_llm_provider_timeout_is_configurable_and_bounded(monkeypatch):
    from unittest.mock import patch

    from mycoder.llm import LLM

    monkeypatch.setenv("MYCODER_LLM_TIMEOUT_SECONDS", "45")
    with patch("mycoder.llm.OpenAI") as openai:
        LLM(model="m", api_key="k")
    assert openai.call_args.kwargs["timeout"] == 45.0

    monkeypatch.setenv("MYCODER_LLM_TIMEOUT_SECONDS", "9999")
    with patch("mycoder.llm.OpenAI") as openai:
        LLM(model="m", api_key="k")
    assert openai.call_args.kwargs["timeout"] == 600.0


def test_llm_rejects_projected_over_budget_call_before_provider(monkeypatch):
    from types import SimpleNamespace

    import pytest
    from structlog.contextvars import bound_contextvars

    from mycoder.llm import LLM
    from mycoder.observability.budget import TokenBudgetExceeded, TokenBudgetGuard
    from mycoder.observability.trace import LLMTracer

    tracer = LLMTracer()
    guard = TokenBudgetGuard(max_tokens_per_session=100, tracer=tracer)
    tracer.register_budget_guard("projected", guard)
    llm = LLM(model="m", api_key="k", tracer=tracer, max_tokens=256)
    called = False

    def _provider(**_kwargs):
        nonlocal called
        called = True
        return iter(())

    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=_provider)))
    with bound_contextvars(session_id="projected"):
        with pytest.raises(TokenBudgetExceeded):
            llm.chat([{"role": "user", "content": "x" * 600}])

    assert called is False


def test_projected_budget_uses_bounded_tool_call_output_margin(monkeypatch):
    from types import SimpleNamespace

    from mycoder.llm import _projected_request_tokens

    monkeypatch.delenv("MYCODER_BUDGET_PROJECTION_OUTPUT_TOKENS", raising=False)
    llm = SimpleNamespace(extra={"max_tokens": 4096})
    messages = [{"role": "user", "content": "inspect and verify the patch"}]

    projected = _projected_request_tokens(llm, messages, None)
    assert projected < estimate_tokens(messages) + 4096
    assert projected >= 1024


# --- Context ---


def test_estimate_tokens():
    msgs = [{"role": "user", "content": "hello world"}]
    t = estimate_tokens(msgs)
    assert t > 0
    assert t < 100


def test_context_snip():
    ctx = ContextManager(max_tokens=3000)
    msgs = [
        {"role": "tool", "tool_call_id": "t1", "content": "x\n" * 1000},
    ]
    before = estimate_tokens(msgs)
    ctx._snip_tool_outputs(msgs)
    after = estimate_tokens(msgs)
    assert after < before


def test_context_compress():
    ctx = ContextManager(max_tokens=2000)
    msgs = []
    for i in range(20):
        msgs.append({"role": "user", "content": f"msg {i} " + "a" * 200})
        msgs.append({"role": "tool", "tool_call_id": f"t{i}", "content": "b" * 2000})
    before = estimate_tokens(msgs)
    ctx.maybe_compress(msgs, None)
    after = estimate_tokens(msgs)
    assert after < before
    assert len(msgs) < 40  # should be compressed


def test_action_compaction_keeps_task_and_recent_tool_pair():
    ctx = ContextManager(max_tokens=32_000)
    messages = [{"role": "user", "content": "fix original bug " + "x" * 2000}]
    for index in range(8):
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": "exploring " + "y" * 500,
                    "tool_calls": [{"id": f"c{index}", "name": "read_file"}],
                },
                {
                    "role": "tool",
                    "tool_call_id": f"c{index}",
                    "content": "evidence " + "z" * 1200,
                },
            ]
        )
    before = estimate_tokens(messages)

    assert ctx.compact_for_action(messages, keep_recent=4) is True
    assert estimate_tokens(messages) < before
    assert "fix original bug" in messages[0]["content"]
    assert messages[-1]["role"] == "tool"
    assert messages[-2]["role"] == "assistant"


def test_action_compaction_preserves_thinking_history_for_tool_rounds():
    """Provider-private reasoning must not be replaced by a prose summary."""
    ctx = ContextManager(max_tokens=32_000)
    messages = [
        {"role": "user", "content": "fix it"},
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "inspect before editing",
            "tool_calls": [{"id": "c1", "name": "read_file"}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "evidence"},
    ]
    messages.extend(
        {"role": "user", "content": f"follow-up {i}"} for i in range(8)
    )

    before = list(messages)
    assert ctx.compact_for_action(messages, keep_recent=2) is False
    assert messages == before


def test_context_compression_does_not_drop_reasoning_content():
    ctx = ContextManager(max_tokens=1000)
    msgs = [
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "think",
            "tool_calls": [{"id": "c1"}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "x" * 3000},
        *({"role": "user", "content": f"turn {i}"} for i in range(12)),
    ]
    before_reasoning = msgs[1]["reasoning_content"]
    ctx.maybe_compress(msgs, None)
    assert any(m.get("reasoning_content") == before_reasoning for m in msgs)


def test_context_compression_keeps_empty_reasoning_field_for_thinking_turn():
    ctx = ContextManager(max_tokens=1000)
    msgs = [
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "",
            "tool_calls": [{"id": "c1"}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "x" * 3000},
        *({"role": "user", "content": f"turn {i}"} for i in range(12)),
    ]
    before = list(msgs)
    ctx.maybe_compress(msgs, None)
    assert any("reasoning_content" in m for m in msgs)
    assert msgs[1] == before[1]


def test_action_compaction_snips_large_one_line_tool_output_without_dropping_reasoning():
    ctx = ContextManager(max_tokens=32_000)
    messages = [
        {"role": "user", "content": "fix it"},
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "inspect before editing",
            "tool_calls": [{"id": "c1", "name": "read_file"}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "x" * 5000},
        {"role": "user", "content": "continue"},
        {"role": "assistant", "content": "still working"},
        {"role": "user", "content": "make the edit"},
    ]
    assert ctx.compact_for_action(messages, keep_recent=2) is True
    assert messages[1]["reasoning_content"] == "inspect before editing"
    assert len(messages[2]["content"]) < 1500


def test_safe_split_never_orphans_a_tool_message():
    """The kept tail must not begin with a 'tool' message - it would be severed
    from the assistant tool_calls that produced it, which the API rejects."""
    ctx = ContextManager(max_tokens=1000)
    messages = [
        {"role": "user", "content": "do it"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "tool_call_id": "c1", "content": "result"},
        {"role": "tool", "tool_call_id": "c2", "content": "result2"},
    ]
    split = ctx._safe_split(messages, keep_recent=1)
    assert messages[split].get("role") != "tool"


def test_compress_never_leaves_an_orphan_tool_reply():
    """After summarisation every tool reply must still follow its tool_calls."""
    ctx = ContextManager(max_tokens=2000)
    msgs = []
    for i in range(20):
        msgs.append({"role": "user", "content": f"msg {i} " + "a" * 200})
        msgs.append({"role": "assistant", "content": None, "tool_calls": [{"id": f"c{i}"}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": "b" * 800})
    ctx.maybe_compress(msgs, None)
    for i, m in enumerate(msgs):
        if m.get("role") == "tool":
            prev = msgs[i - 1]
            assert prev.get("role") == "tool" or prev.get("tool_calls"), f"orphan tool at {i}"


# --- Session ---


def test_session_save_load(tmp_path, monkeypatch):
    monkeypatch.setattr(session_module, "SESSIONS_DIR", tmp_path)
    msgs = [{"role": "user", "content": "test message"}]
    save_session(msgs, "test-model", "pytest_test_session")
    loaded = load_session("pytest_test_session")
    assert loaded is not None
    assert loaded[0] == msgs
    assert loaded[1] == "test-model"


def test_session_name_is_sanitized(tmp_path, monkeypatch):
    monkeypatch.setattr(session_module, "SESSIONS_DIR", tmp_path)
    msgs = [{"role": "user", "content": "test message"}]
    sid = save_session(msgs, "test-model", "../Research Notes!")

    assert sid == "Research-Notes"
    assert (tmp_path / "Research-Notes.json").exists()
    assert load_session("../Research Notes!") is not None


def test_session_not_found():
    assert load_session("nonexistent_session_id") is None


def test_list_sessions():
    sessions = list_sessions()
    assert isinstance(sessions, list)


# --- Cost estimation ---


def test_cost_estimation_known_model():
    from mycoder.llm import LLM

    llm = LLM.__new__(LLM)
    llm.model = "gpt-5.4"
    llm.total_prompt_tokens = 1_000_000
    llm.total_completion_tokens = 500_000
    cost = llm.estimated_cost
    assert cost is not None
    assert cost == 2.5 + 7.5  # $2.5/M in + $15/M out * 0.5M


def test_cost_estimation_unknown_model():
    from mycoder.llm import LLM

    llm = LLM.__new__(LLM)
    llm.model = "some-custom-model"
    llm.total_prompt_tokens = 1000
    llm.total_completion_tokens = 500
    assert llm.estimated_cost is None


# --- Changed files tracking ---


def test_edit_tracks_changed_files(tmp_path):
    from mycoder.tools.edit import _changed_files

    _changed_files.clear()
    edit = get_tool("edit_file")
    path = tmp_path / "sample.py"
    path.write_text("aaa\nbbb\n")
    edit.execute(file_path=str(path), old_string="aaa", new_string="zzz")
    assert any(str(path) in p for p in _changed_files)
    _changed_files.clear()


def test_write_tracks_changed_files(tmp_path):
    from mycoder.tools.edit import _changed_files

    _changed_files.clear()
    write = get_tool("write_file")
    path = tmp_path / "tracked.txt"
    write.execute(file_path=str(path), content="tracked\n")
    assert any(path.name in p for p in _changed_files)
    _changed_files.clear()


# --- Agent tool execution ---


def test_agent_tool_scope_is_per_instance():
    """An Agent restricted to a subset of tools must not resolve tools outside it."""
    only_read = [get_tool("read_file")]
    agent = Agent(llm=LLM.__new__(LLM), tools=only_read)
    assert set(agent._tool_by_name) == {"read_file"}

    class _TC:
        name = "bash"  # a real, registered tool - but not in this agent's set
        id = "x"
        arguments = {"command": "echo hi"}

    assert "unknown tool 'bash'" in agent._exec_tool(_TC())


def test_exec_tool_distinguishes_bad_args_from_internal_error():
    """A TypeError raised inside a tool must not be reported as bad arguments."""
    from mycoder.tools.base import Tool

    class _Boom(Tool):
        name = "boom"
        description = "raises TypeError internally"
        parameters = {"type": "object", "properties": {}, "required": []}

        def execute(self):
            raise TypeError("internal explosion")

    agent = Agent(llm=LLM.__new__(LLM), tools=[_Boom()])

    class _BadArgs:
        name, id, arguments = "boom", "1", {"unexpected": 1}

    class _Good:
        name, id, arguments = "boom", "2", {}

    assert "bad arguments" in agent._exec_tool(_BadArgs())
    assert "Error executing boom" in agent._exec_tool(_Good())
    assert "bad arguments" not in agent._exec_tool(_Good())


def test_interrupt_backfills_missing_tool_replies():
    """A half-finished tool round must be repaired so history stays valid."""
    agent = Agent(llm=LLM.__new__(LLM), tools=[])
    agent.messages = [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "a"}, {"id": "b"}]},
        {"role": "tool", "tool_call_id": "a", "content": "done"},
    ]

    class _TC:
        def __init__(self, i):
            self.id = i

    agent._answer_pending_tool_calls([_TC("a"), _TC("b")])
    replies = [m for m in agent.messages if m.get("role") == "tool"]
    ids = [m["tool_call_id"] for m in replies]
    assert sorted(ids) == ["a", "b"]
    assert ids.count("a") == 1  # the already-answered call wasn't duplicated


def test_compression_stats_accounts_saved_tokens():
    """P1: compression accounting — tokens saved / avg compression ratio."""
    from mycoder.context import ContextManager

    ctx = ContextManager(max_tokens=2000)
    msgs = [{"role": "user", "content": f"m{i} " + "a" * 300} for i in range(20)]
    assert ctx.maybe_compress(msgs, None)  # summarization fired
    s = ctx.compression_stats()
    assert s["compressions"] == 1
    assert s["tokens_before"] > s["tokens_after"]
    assert s["tokens_saved"] > 0
    assert s["avg_compression_ratio"] > 0.0


def test_predictive_executor_fires_when_tool_args_complete():
    """Predictive execution: a streamed tool call is handed to the executor the
    moment its args parse as complete JSON — before generation finishes."""
    from mycoder.llm import LLM

    class _F:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    def fake_stream(params):
        tc1 = _F(index=0, id="t1", function=_F(name="read_file", arguments='{"file_'))
        yield _F(choices=[_F(delta=_F(content=None, tool_calls=[tc1]))], usage=None)
        tc2 = _F(index=0, id=None, function=_F(name=None, arguments='path": "a.py"}'))
        yield _F(choices=[_F(delta=_F(content=None, tool_calls=[tc2]))], usage=None)

    llm = LLM(model="m", api_key="k", base_url="http://localhost")
    llm._call_with_retry = lambda params: fake_stream(params)

    executed = []
    resp = llm.chat([{"role": "user", "content": "x"}], predictive_executor=executed.append)
    assert len(executed) == 1  # fired once, on completion of the args
    assert executed[0].name == "read_file"
    assert executed[0].arguments == {"file_path": "a.py"}
    assert resp.tool_calls[0].name == "read_file"
    assert resp.tool_calls[0].arguments == {"file_path": "a.py"}


def test_agent_predicts_only_explicitly_safe_tools(monkeypatch):
    """Speculation must never execute writes or unclassified third-party tools."""
    import threading

    from mycoder.agent import Agent
    from mycoder.llm import LLMResponse, ToolCall
    from mycoder.tools.base import Tool

    calls: list[str] = []
    read_started = threading.Event()

    class _Read(Tool):
        name = "safe_read"
        description = "read"
        parameters = {"type": "object", "properties": {}}
        predictive_safe = True

        def execute(self):
            calls.append(self.name)
            read_started.set()
            return "read"

    class _Write(Tool):
        name = "write"
        description = "write"
        parameters = {"type": "object", "properties": {}}
        idempotent = True

        def execute(self):
            calls.append(self.name)
            return "written"

    tool_calls = [
        ToolCall(id="r", name="safe_read", arguments={}),
        ToolCall(id="w", name="write", arguments={}),
    ]

    class _LLM:
        def __init__(self):
            self.round = 0

        def chat(self, *, predictive_executor=None, **_kwargs):
            self.round += 1
            if self.round == 1:
                for call in tool_calls:
                    predictive_executor(call)
                assert read_started.wait(1)
                assert calls == ["safe_read"]
                return LLMResponse(tool_calls=tool_calls)
            return LLMResponse(content="done")

    monkeypatch.setenv("MYCODER_INJECTION_GUARD", "off")
    agent = Agent(llm=_LLM(), tools=[_Read(), _Write()])
    assert agent.chat("go") == "done"
    assert calls == ["safe_read", "write"]


def test_openrouter_reasoning_tokens_are_read_from_final_stream_chunk():
    """OpenRouter reports reasoning usage in completion_tokens_details."""
    from mycoder.llm import LLM

    class _F:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    usage = _F(
        prompt_tokens=10,
        completion_tokens=7,
        completion_tokens_details=_F(reasoning_tokens=4),
    )

    def fake_stream(_params):
        yield _F(choices=[_F(delta=_F(content="three", tool_calls=None))], usage=None)
        yield _F(choices=[], usage=usage)

    llm = LLM(
        model="minimax/minimax-m3:free",
        api_key="k",
        provider="openrouter",
    )
    llm._call_with_retry = fake_stream

    response = llm.chat([{"role": "user", "content": "strawberry"}])

    assert response.reasoning_tokens == 4
    assert response.completion_tokens == 7
    assert llm.total_reasoning_tokens == 4


def test_required_tool_choice_is_forwarded_and_provider_fallback_is_bounded():
    import httpx
    from openai import BadRequestError

    from mycoder.llm import LLM

    calls = []

    def fake_call(params):
        calls.append(dict(params))
        if len(calls) < 3:
            raise BadRequestError(
                "unsupported parameter",
                response=httpx.Response(
                    400,
                    request=httpx.Request("POST", "https://provider.invalid/chat"),
                ),
                body=None,
            )
        return iter(())

    llm = LLM(model="deepseek-flash", api_key="k", provider="openrouter")
    llm._call_with_retry = fake_call
    tools = [{"type": "function", "function": {"name": "edit_file"}}]

    llm.chat(
        [{"role": "user", "content": "edit now"}],
        tools=tools,
        tool_choice="required",
    )

    assert calls[0]["tool_choice"] == "required"
    assert "stream_options" not in calls[1]
    assert calls[1]["tool_choice"] == "required"
    assert "tool_choice" not in calls[2]
    assert len(calls) == 3


def test_strict_tool_choice_never_silently_degrades():
    import httpx
    from openai import BadRequestError

    from mycoder.llm import LLM, ToolChoiceCapabilityError

    calls = []

    def fake_call(params):
        calls.append(dict(params))
        raise BadRequestError(
            "unsupported tool choice",
            response=httpx.Response(
                400,
                request=httpx.Request("POST", "https://provider.invalid/chat"),
            ),
            body=None,
        )

    llm = LLM(model="deepseek-flash", api_key="k", provider="openrouter")
    llm._call_with_retry = fake_call
    with pytest.raises(ToolChoiceCapabilityError):
        llm.chat(
            [{"role": "user", "content": "edit now"}],
            tools=[{"type": "function", "function": {"name": "edit_file"}}],
            tool_choice="required",
            strict_tool_choice=True,
        )
    assert len(calls) == 2
    assert all("tool_choice" in call for call in calls)


def test_provider_request_timeout_and_retry_budget_are_forwarded():
    from mycoder.llm import LLM

    observed = {}

    def fake_call(params, max_retries=3):
        observed["params"] = dict(params)
        observed["max_retries"] = max_retries
        return iter(())

    llm = LLM(model="deepseek-flash", api_key="k", provider="deepseek")
    llm._call_with_retry = fake_call

    llm.chat(
        [{"role": "user", "content": "plan"}],
        timeout_seconds=12.5,
        request_max_retries=1,
    )

    assert observed["params"]["timeout"] == 12.5
    assert observed["max_retries"] == 1


def test_deepseek_reasoning_content_is_preserved_for_next_tool_round():
    """Thinking-mode tool calls must replay their reasoning on the next request."""
    from mycoder.llm import LLM

    class _F:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    tool_delta = _F(
        index=0,
        id="call_1",
        function=_F(name="read_file", arguments='{"path":"a.py"}'),
    )

    def fake_stream(_params):
        yield _F(
            choices=[
                _F(
                    delta=_F(
                        content=None,
                        tool_calls=[tool_delta],
                        model_extra={"reasoning_content": "inspect the file first"},
                    )
                )
            ],
            usage=None,
        )

    llm = LLM(model="deepseek-flash", api_key="k", provider="deepseek")
    llm._call_with_retry = fake_stream

    response = llm.chat(
        [{"role": "user", "content": "fix it"}],
        tools=[{"type": "function", "function": {"name": "read_file"}}],
    )

    assert response.reasoning_content == "inspect the file first"
    assert response.message["reasoning_content"] == "inspect the file first"
    assert response.message["tool_calls"][0]["function"]["name"] == "read_file"


def test_deepseek_dsml_tool_call_is_parsed_from_reasoning_channel():
    from mycoder.llm import LLM

    class _F:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    dsml = """<｜｜DSML｜｜ invoke name="edit_file">
<｜｜DSML｜｜ parameter name="file_path" string="true">module.py</｜｜DSML｜｜ parameter>
<｜｜DSML｜｜ parameter name="old_string" string="true">VALUE = 1</｜｜DSML｜｜ parameter>
<｜｜DSML｜｜ parameter name="new_string" string="true">VALUE = 2</｜｜DSML｜｜ parameter>
</｜｜DSML｜｜ invoke>"""

    def fake_stream(_params):
        yield _F(
            choices=[
                _F(
                    delta=_F(
                        content="I will apply the edit.",
                        tool_calls=None,
                        model_extra={"reasoning_content": dsml},
                    )
                )
            ],
            usage=None,
        )

    llm = LLM(model="deepseek-flash", api_key="k", provider="deepseek")
    llm._call_with_retry = fake_stream

    response = llm.chat(
        [{"role": "user", "content": "fix it"}],
        tools=[{"type": "function", "function": {"name": "edit_file"}}],
    )

    assert response.tool_calls[0].name == "edit_file"
    assert response.tool_calls[0].arguments == {
        "file_path": "module.py",
        "old_string": "VALUE = 1",
        "new_string": "VALUE = 2",
    }
