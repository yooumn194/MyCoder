"""接线验证 — 5 个已建组件确实接进了生产默认路径。

Each test pins the "wired by default" claim from docs/project-gaps-v2.md:
these components exist AND the production default path (CLI/API/tools) uses
them — not just that the module exists with a passing unit test.
"""

from unittest import mock

from mycoder.tools import ALL_TOOLS
from mycoder.tools.memory_tools import MemorySearchTool
from mycoder.tools.selector import ToolSelector


class _FakeLLM:
    """Matches the mycoder.llm constructor signature (model, api_key,
    base_url, *, tracer, caller, **kwargs) so build_model_factory can rebuild
    it for a tier model."""

    def __init__(
        self,
        model="std-model",
        api_key="k",
        base_url="http://x",
        *,
        tracer=None,
        caller="llm",
        **kwargs,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self._tracer = tracer
        self.caller = caller
        self.extra = kwargs or {"temperature": 0.2}
        self.client = None


# ------------------------------------------------- 1. tool_selector (CLI)
def test_agent_with_tool_selector_injects_subset_not_all():
    """The default CLI wiring (tool_selector=ToolSelector()) injects a
    relevance-filtered subset instead of all 20 tool schemas every turn."""
    from mycoder.agent import Agent

    agent = Agent(llm=mock.Mock(), tools=ALL_TOOLS, tool_selector=ToolSelector())
    names = {s["function"]["name"] for s in agent._tool_schemas("search memory retrieve")}
    assert len(names) < len(ALL_TOOLS)  # subset, not everything
    assert "memory_search" in names     # the relevant tool made the cut


# ------------------------------------------------- 2. model_factory (CLI)
def test_subagent_tool_orchestrator_carries_model_factory():
    """SpawnSubagentTool's lazily-built orchestrator routes sub-agents to a
    tier-appropriate model (CLI production path)."""
    from mycoder.tools.subagent_tools import SpawnSubagentTool

    parent = mock.Mock()
    parent.llm = _FakeLLM(model="std-model")
    parent.tools = ALL_TOOLS
    tool = SpawnSubagentTool()
    tool._parent_agent = parent
    orch = tool._get_orchestrator()
    assert orch is not None
    assert orch.model_factory is not None
    # factory("fast") returns a tier model (or None when it equals the base)
    tier_model = orch.model_factory("fast")
    assert tier_model is None or tier_model.model != "std-model"


# ------------------------------------------------- 3. model_factory (API)
def test_api_orchestrator_builder_carries_model_factory():
    """api/dependencies.get_orchestrator's builder wires model_factory so API
    sub-agents get tier-appropriate models too — and re-planning experience
    settles to the memory DB."""
    from api.dependencies import get_orchestrator

    builder = get_orchestrator(state_backend=mock.Mock())
    orch = builder("session-x", llm=_FakeLLM(model="std-model"))
    assert orch.model_factory is not None
    assert orch._experience_store is not None  # #8: API re-plan experience lands


# ------------------------------------------------- 4. agentic RAG (default)
def test_memory_search_tool_agentic_by_default():
    """MemorySearchTool's default retriever is the agentic loop (retrieve ->
    judge -> re-query), not single-shot; a plain retriever can be forced or
    injected."""
    from mycoder.memory.agentic import AgenticRetriever
    from mycoder.memory.retriever import HybridRetriever

    default = MemorySearchTool()
    assert isinstance(default._retriever(), AgenticRetriever)

    plain = MemorySearchTool(agentic=False)
    assert isinstance(plain._retriever(), HybridRetriever)

    injected = MemorySearchTool(retriever=HybridRetriever(mock.Mock()))
    assert isinstance(injected._retriever(), HybridRetriever)


# ------------------------------------------------- 5. re-plan experience (CLI)
def test_agent_wires_replan_experience_store():
    """Agent(experience_store=...) reaches the subagent orchestrator, so CLI
    deviation playbooks persist to the memory DB (对标 Hermes 经验沉淀)."""
    from mycoder.agent import Agent
    from mycoder.llm import LLM
    from mycoder.tools.subagent_tools import SpawnSubagentTool

    store = mock.Mock()
    agent = Agent(llm=LLM.__new__(LLM), tools=ALL_TOOLS, experience_store=store)
    tool = next(t for t in agent.tools if isinstance(t, SpawnSubagentTool))
    assert tool._experience_store == store
    orch = tool._get_orchestrator()
    assert orch._experience_store == store
    # #8: the CLI orchestrator now carries a planner too, so GOAL_DRIFT
    # re-planning actually fires (previously it was single-explorer only).
    assert orch._planner is not None


def test_agent_wires_subagent_budget_guard():
    """Agent(budget_guard=...) reaches the subagent orchestrator so CLI
    sub-agents get per-session token budget protection (#10)."""
    from mycoder.agent import Agent
    from mycoder.llm import LLM
    from mycoder.tools.subagent_tools import SpawnSubagentTool

    bg = mock.Mock()
    agent = Agent(llm=LLM.__new__(LLM), tools=ALL_TOOLS, budget_guard=bg)
    tool = next(t for t in agent.tools if isinstance(t, SpawnSubagentTool))
    assert tool._budget_guard == bg
    orch = tool._get_orchestrator()
    assert orch._budget_guard == bg


def test_parallel_dedup_executes_duplicate_idempotent_once():
    """#12: two parallel identical idempotent calls execute ONCE; the result
    is broadcast — otherwise both threads miss the idempotency cache and
    re-apply the side effect."""
    from mycoder.agent import Agent
    from mycoder.llm import LLM
    from mycoder.tools.base import Tool

    calls: list[tuple] = []

    class _Write(Tool):
        name = "write_file"
        idempotent = True
        description = "write"
        parameters = {"type": "object", "properties": {}}

        def execute(self, path, content):
            calls.append((path, content))
            return f"wrote {path}"

    class _TC:
        def __init__(self, name, args, tid):
            self.name = name
            self.arguments = args
            self.id = tid

    agent = Agent(llm=LLM.__new__(LLM), tools=[_Write()])
    tcs = [
        _TC("write_file", {"path": "a.txt", "content": "x"}, "1"),
        _TC("write_file", {"path": "a.txt", "content": "x"}, "2"),  # duplicate
    ]
    results = agent._exec_tools_parallel(tcs)
    assert len(calls) == 1  # duplicate ran once
    assert results == ["wrote a.txt", "wrote a.txt"]  # result broadcast


def test_parallel_keeps_distinct_calls_distinct():
    """#12: non-identical calls (or non-idempotent ones) still run individually."""
    from mycoder.agent import Agent
    from mycoder.llm import LLM
    from mycoder.tools.base import Tool

    calls: list[tuple] = []

    class _Echo(Tool):
        name = "fetch_url"
        idempotent = False
        description = "fetch"
        parameters = {"type": "object", "properties": {}}

        def execute(self, url):
            calls.append(url)
            return url

    class _TC:
        def __init__(self, name, args, tid):
            self.name = name
            self.arguments = args
            self.id = tid

    agent = Agent(llm=LLM.__new__(LLM), tools=[_Echo()])
    tcs = [
        _TC("fetch_url", {"url": "http://a"}, "1"),
        _TC("fetch_url", {"url": "http://a"}, "2"),  # non-idempotent: both run
    ]
    results = agent._exec_tools_parallel(tcs)
    assert len(calls) == 2  # non-idempotent duplicates are NOT deduped
    assert results == ["http://a", "http://a"]


# ------------------------------------------------- 6. remaining low-item fixes
def test_budget_guard_default_session_budget(monkeypatch):
    """#18: the default session budget is 100k (cumulative), NOT the 4096
    per-response cap that silently killed API runs without max_tokens."""
    from mycoder.observability.budget import TokenBudgetGuard

    monkeypatch.delenv("MYCODER_SESSION_BUDGET", raising=False)
    assert TokenBudgetGuard().max_tokens_per_session == 100_000

    monkeypatch.setenv("MYCODER_SESSION_BUDGET", "5000")
    assert TokenBudgetGuard().max_tokens_per_session == 5000


def test_selection_query_grows_with_conversation():
    """#13: the tool-selection query reflects the RECENT conversation (last
    user message + tool names), so the injected tool set can grow per turn."""
    from mycoder.agent import Agent
    from mycoder.llm import LLM

    agent = Agent(llm=LLM.__new__(LLM))
    agent.messages = [
        {"role": "user", "content": "搜索文件"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "name": "grep_search"}]},
        {"role": "tool", "tool_call_id": "1", "content": "结果"},
        {"role": "user", "content": "现在抓取一个 URL"},
    ]
    q = agent._selection_query()
    assert "抓取" in q            # the latest user message drives selection
    assert "grep_search" in q     # tool names surface too
