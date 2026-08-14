"""P0 tool idempotency: retry_safe + IdempotencyStore + Agent dedup."""

from types import SimpleNamespace

import pytest

from mycoder.agent import Agent
from mycoder.llm import LLM
from mycoder.tools.base import Tool
from mycoder.tools.correction import run_with_correction
from mycoder.tools.idempotency import IdempotencyStore


def _tc(tool: Tool, arguments: dict):
    return SimpleNamespace(name=tool.name, arguments=arguments, id="c1")


class _IdemTool(Tool):
    name = "dummy"
    description = "dummy idempotent tool"
    parameters = {"type": "object", "properties": {}, "required": []}

    def __init__(self) -> None:
        self.calls = 0

    def execute(self, **kwargs) -> str:
        self.calls += 1
        return f"result-{self.calls}"


class _WriteTool(Tool):
    name = "writer"
    description = "side-effecting write"
    parameters = {"type": "object", "properties": {}, "required": []}
    idempotent = False

    def __init__(self) -> None:
        self.calls = 0

    def execute(self, **kwargs) -> str:
        self.calls += 1
        return f"wrote-{self.calls}"


# ------------------------------------------------------------ retry_safe
def test_retry_safe_false_does_not_auto_retry():
    calls = {"n": 0}

    def _flaky():
        calls["n"] += 1
        raise TimeoutError("transient")

    with pytest.raises(TimeoutError):
        run_with_correction(_flaky, retry_safe=False, sleep_fn=lambda _: None)
    assert calls["n"] == 1  # a non-idempotent tool's failure surfaces immediately


def test_retry_safe_true_still_retries():
    calls = {"n": 0}

    def _flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError("transient")
        return "ok"

    assert run_with_correction(_flaky, retry_safe=True, sleep_fn=lambda _: None) == "ok"
    assert calls["n"] == 3


# ------------------------------------------------------------ store
def test_idempotency_store_key_and_dedup():
    store = IdempotencyStore()
    key = store.key("write_file", {"file_path": "a.py", "content": "x"})
    assert key not in store
    store.put(key, "Wrote 1 lines")
    assert key in store
    # same args -> same key, cached result returned
    assert store.get(store.key("write_file", {"content": "x", "file_path": "a.py"})) == "Wrote 1 lines"
    # different args -> different key
    assert store.key("write_file", {"file_path": "a.py", "content": "y"}) != key


# ------------------------------------------------------------ agent wiring
def test_agent_serves_identical_idempotent_call_from_cache():
    tool = _IdemTool()
    agent = Agent(llm=LLM.__new__(LLM), tools=[tool])
    first = agent._exec_tool(_tc(tool, {"a": 1}))
    second = agent._exec_tool(_tc(tool, {"a": 1}))
    assert first == "result-1" and second == "result-1"
    assert tool.calls == 1  # re-issuing an identical idempotent call does NOT re-execute


def test_agent_never_caches_non_idempotent_tool():
    tool = _WriteTool()
    agent = Agent(llm=LLM.__new__(LLM), tools=[tool])
    first = agent._exec_tool(_tc(tool, {}))
    second = agent._exec_tool(_tc(tool, {}))
    assert first == "wrote-1" and second == "wrote-2"
    assert tool.calls == 2  # side-effecting tool always re-executes


def test_idempotency_store_hit_rate():
    """#40: cache hit/miss counters — the '缓存命中率' metric source."""
    store = IdempotencyStore()
    key = store.key("write_file", {"path": "a.txt"})
    assert store.get(key) is None  # miss
    store.put(key, "ok")
    assert store.get(key) == "ok"  # hit
    store.put(key, "ok")
    assert store.get(key) == "ok"  # hit again
    s = store.stats()
    assert s["hits"] == 2
    assert s["misses"] == 1
    assert s["hit_rate"] == round(2 / 3, 4)
    # clear resets counters too
    store.clear()
    assert store.stats()["total"] == 0


def test_agent_tool_metrics_success_and_failure():
    """Tool success / failure counters -> success_rate / failure_rate (面经)."""

    class _OkTool(Tool):
        name = "ok_tool"
        idempotent = True
        description = "ok"
        parameters = {"type": "object", "properties": {}}

        def execute(self, **kw):
            return "fine"

    class _ErrTool(Tool):
        name = "err_tool"
        idempotent = True
        description = "err"
        parameters = {"type": "object", "properties": {}}

        def execute(self, **kw):
            return "Error: boom"

    agent = Agent(llm=LLM.__new__(LLM), tools=[_OkTool(), _ErrTool()])
    agent._exec_tool(_tc(_OkTool(), {}))   # success
    agent._exec_tool(_tc(_ErrTool(), {}))  # failure (Error-string return)

    m = agent._tool_metrics()
    assert m["calls"] == 2
    assert m["successes"] == 1
    assert m["failures"] == 1
    assert m["success_rate"] == 0.5
    assert m["failure_rate"] == 0.5
    assert m["retries"] == 0


def test_idempotency_store_bounded_evicts_oldest():
    """The cache is bounded (FIFO) so a long session can't grow without bound."""
    store = IdempotencyStore(maxsize=2)
    store.put(store.key("a", {"x": 1}), "1")
    store.put(store.key("b", {"x": 1}), "2")
    store.put(store.key("c", {"x": 1}), "3")  # evicts "a"
    assert store.get(store.key("a", {"x": 1})) is None  # evicted
    assert store.get(store.key("b", {"x": 1})) == "2"
    assert store.get(store.key("c", {"x": 1})) == "3"
    s = store.stats()
    assert s["misses"] == 1 and s["hits"] == 2 and s["total"] == 3
    assert s["hit_rate"] == round(2 / 3, 4)
