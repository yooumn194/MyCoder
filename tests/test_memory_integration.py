"""Phase 5 integration: planning_guard injection, Self-Correction pattern
settlement, plan.json decision distillation, token budget (spec: 4 tests)."""

import mycoder.planner as planner_mod
from mycoder.memory import MemoryEntry
from mycoder.memory.integration import MemoryIntegration
from mycoder.memory.prompt import MemoryPromptInjector
from mycoder.planner import (
    PlanStore,
    TaskPlan,
    TodoItem,
    TodoStatus,
    get_pending_memory_section,
    set_active_plan,
)
from mycoder.tools.correction import run_with_correction
from mycoder.tools.todo_tools import TodoUpdateTool, TodoWriteTool


def test_planning_guard_injects_memory_into_todo_write(
    memory_store, memory_retriever, tmp_path
):
    memory_store.save(MemoryEntry(content="用户偏好：认证模块使用JWT", type="user"))
    MemoryIntegration(store=memory_store, retriever=memory_retriever).install()

    # guard never blocks todo_write, but stages a memory section as a side channel
    assert planner_mod.planning_guard("todo_write", query="认证模块") is None
    section = get_pending_memory_section()
    assert "认证模块使用JWT" in section

    # todo_write prepends the staged section and consumes it
    tool = TodoWriteTool(store=PlanStore(base_dir=tmp_path / "plans"))
    out = tool.execute(
        task_goal="重构认证模块",
        todos=[{"id": "s1", "description": "重构认证逻辑"}],
    )
    assert "认证模块使用JWT" in out
    assert get_pending_memory_section() == ""

    # guard with no query stages nothing
    assert planner_mod.planning_guard("todo_write") is None
    assert get_pending_memory_section() == ""


def test_correction_recovery_settles_pattern(memory_store, memory_retriever):
    MemoryIntegration(store=memory_store, retriever=memory_retriever).install()

    calls = {"n": 0}

    def _flaky(x):
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError("transient network error")
        return x * 2

    assert run_with_correction(_flaky, x=1, sleep_fn=lambda _: None) == 2
    patterns = memory_store.list(type="pattern")
    assert len(patterns) == 1
    assert "重试" in patterns[0].content
    assert patterns[0].type == "pattern"


def test_plan_completion_settles_decision(memory_store, memory_retriever, tmp_path):
    MemoryIntegration(store=memory_store, retriever=memory_retriever).install()

    plan = TaskPlan(
        goal="修复登录超时",
        items=[
            TodoItem(id="s1", description="定位根因"),
            TodoItem(id="s2", description="实施修复"),
        ],
    )
    pstore = PlanStore(base_dir=tmp_path / "plans")
    pstore.save(plan)
    set_active_plan(plan)
    tool = TodoUpdateTool(store=pstore)

    tool.execute(step_id="s1", status=TodoStatus.DONE)
    assert memory_store.list(type="decision") == []  # plan not complete yet

    tool.execute(step_id="s2", status=TodoStatus.DONE)
    decisions = memory_store.list(type="decision")
    assert len(decisions) == 1
    assert "修复登录超时" in decisions[0].content
    set_active_plan(None)


def test_injector_respects_token_budget(tmp_path):
    # BM25-only store (embedder=None): 20 distinct memories, no vector dedup
    from mycoder.memory import HybridRetriever, MemoryStore

    store = MemoryStore(
        project_dir=tmp_path / "proj", global_dir=tmp_path / "glob", embedder=None
    )
    retriever = HybridRetriever(store)
    for i in range(20):
        store.save(MemoryEntry(content=f"偏好记录{i}：项目使用Python构建系统，编号{i}"))
    budgeted = MemoryPromptInjector(retriever, max_tokens=100).build_memory_section(
        "Python 构建"
    )
    unlimited = MemoryPromptInjector(retriever, max_tokens=10 ** 6).build_memory_section(
        "Python 构建"
    )
    assert budgeted  # at least something is injected
    assert len(budgeted) < len(unlimited)  # budget truncates the section


def test_settlement_failures_never_break_tools():
    """A broken store must not raise out of the memory hooks."""

    class _BrokenStore:
        vector_backend_name = "broken"

        def save(self, *a, **k):
            raise RuntimeError("disk full")

    from mycoder.memory.integration import MemoryIntegration

    MemoryIntegration(store=_BrokenStore(), retriever=object()).install()

    calls = {"n": 0}

    def _flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise TimeoutError("transient")
        return "ok"

    # the recovery hook tries to settle a PatternMemory, the store raises, and
    # the hook swallows it — the tool result must come through untouched
    assert run_with_correction(_flaky, sleep_fn=lambda _: None) == "ok"
