"""Tests for core modules: config, context, session, imports."""

from mycoder import Agent, LLM, Config, ALL_TOOLS, __version__
from mycoder import session as session_module
from mycoder.context import ContextManager, estimate_tokens
from mycoder.session import save_session, load_session, list_sessions
from mycoder.tools import get_tool


def test_version():
    assert __version__ == "0.4.0"


def test_public_api_exports():
    """Users should be able to import key classes from the top-level package."""
    assert Agent is not None
    assert LLM is not None
    assert Config is not None
    # bash was replaced by execute_in_sandbox; sync_workspace, grep_search,
    # list_files, fetch_url, and the Phase 3 planning tools added
    assert {t.name for t in ALL_TOOLS} == {
        "execute_in_sandbox",
        "sync_workspace",
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


def test_openrouter_config_from_provider_env(monkeypatch):
    monkeypatch.setenv("MYCODER_PROVIDER", "OpenRouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")

    c = Config.from_env()

    assert c.provider == "openrouter"
    assert c.api_key == "or-key"
    assert c.base_url == "https://openrouter.ai/api/v1"
    assert c.model == "minimax/minimax-m3:free"


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

    assert c.model == "deepseek-chat"


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
    assert c.model == "deepseek-chat"


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
