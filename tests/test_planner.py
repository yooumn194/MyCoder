"""Tests for the Phase 3 planning engine (planner + todo tools + guard)."""

import pytest

from corecoder.planner import (
    PlanStore,
    PlanValidator,
    TaskPlan,
    TodoItem,
    TodoStatus,
    clear_active_plan,
    planning_guard,
    set_active_plan,
)
from corecoder.tools.todo_tools import TodoUpdateTool, TodoWriteTool


@pytest.fixture(autouse=True)
def _clean_planner():
    """No active plan leaks between tests."""
    clear_active_plan()
    yield
    clear_active_plan()


# ---------------------------------------------------------------------------
# PlanValidator
# ---------------------------------------------------------------------------

def test_validator_accepts_valid_plan():
    todos = [
        TodoItem(id="s1", description="search"),
        TodoItem(id="s2", description="edit", depends_on=["s1"]),
    ]
    assert PlanValidator.validate(todos) is None


def test_validator_detects_cycle():
    todos = [
        TodoItem(id="a", description="a", depends_on=["b"]),
        TodoItem(id="b", description="b", depends_on=["a"]),
    ]
    msg = PlanValidator.validate(todos)
    assert msg and msg.startswith("❌") and "循环依赖" in msg


def test_validator_detects_dangling_dependency():
    todos = [TodoItem(id="s1", description="s", depends_on=["ghost"])]
    msg = PlanValidator.validate(todos)
    assert msg and "依赖了不存在的步骤" in msg


def test_validator_warns_on_granularity():
    todos = [TodoItem(id=f"s{i}", description=str(i)) for i in range(11)]
    msg = PlanValidator.validate(todos)
    assert msg and msg.startswith("⚠️")


# ---------------------------------------------------------------------------
# TodoWriteTool / TodoUpdateTool
# ---------------------------------------------------------------------------

def test_todo_write_creates_plan(tmp_path):
    tool = TodoWriteTool(store=PlanStore(base_dir=tmp_path))
    r = tool.execute(
        task_goal="refactor auth",
        todos=[{"id": "s1", "description": "search"}, {"id": "s2", "description": "edit"}],
    )
    assert "✅ 计划已创建：2 步" in r
    assert PlanStore(base_dir=tmp_path).load() is not None


def test_todo_write_rejects_cyclic_plan(tmp_path):
    tool = TodoWriteTool(store=PlanStore(base_dir=tmp_path))
    r = tool.execute(
        task_goal="x",
        todos=[
            {"id": "a", "description": "a", "depends_on": ["b"]},
            {"id": "b", "description": "b", "depends_on": ["a"]},
        ],
    )
    assert r.startswith("❌")  # hard rejection, nothing written
    assert PlanStore(base_dir=tmp_path).load() is None


def test_todo_write_rejects_dangling_plan(tmp_path):
    tool = TodoWriteTool(store=PlanStore(base_dir=tmp_path))
    r = tool.execute(
        task_goal="x",
        todos=[{"id": "s1", "description": "s", "depends_on": ["ghost"]}],
    )
    assert r.startswith("❌")


def test_todo_write_warns_but_writes_oversized_plan(tmp_path):
    tool = TodoWriteTool(store=PlanStore(base_dir=tmp_path))
    r = tool.execute(
        task_goal="x",
        todos=[{"id": f"s{i}", "description": str(i)} for i in range(11)],
    )
    assert "✅" in r and "⚠️" in r  # written, with the granularity warning


def test_todo_update_marks_steps(tmp_path):
    write = TodoWriteTool(store=PlanStore(base_dir=tmp_path))
    update = TodoUpdateTool(store=PlanStore(base_dir=tmp_path))
    write.execute(task_goal="g", todos=[{"id": "s1", "description": "d"}])
    assert "in_progress" in update.execute(step_id="s1", status="in_progress")
    assert "done" in update.execute(step_id="s1", status="done")


def test_todo_update_unknown_step(tmp_path):
    write = TodoWriteTool(store=PlanStore(base_dir=tmp_path))
    update = TodoUpdateTool(store=PlanStore(base_dir=tmp_path))
    write.execute(task_goal="g", todos=[{"id": "s1", "description": "d"}])
    assert "不存在" in update.execute(step_id="nope", status="done")


def test_todo_update_without_plan(tmp_path):
    update = TodoUpdateTool(store=PlanStore(base_dir=tmp_path))
    assert "无活跃计划" in update.execute(step_id="s1", status="done")


# ---------------------------------------------------------------------------
# planning_guard (soft constraint + hard interception)
# ---------------------------------------------------------------------------

def test_guard_always_allows_exploration_and_planning():
    clear_active_plan()
    assert planning_guard("grep_search") is None
    assert planning_guard("read_file") is None
    assert planning_guard("list_files") is None
    assert planning_guard("todo_write") is None
    assert planning_guard("todo_update") is None


def test_guard_soft_mode_allows_mutation_without_plan(monkeypatch):
    """No plan + enforcement off -> open-ended execution (Phase 1/2 preserved)."""
    clear_active_plan()
    monkeypatch.delenv("CORECODER_ENFORCE_PLANNING", raising=False)
    assert planning_guard("write_file") is None
    assert planning_guard("edit_file") is None


def test_guard_hard_mode_blocks_mutation_without_plan(monkeypatch):
    clear_active_plan()
    monkeypatch.setenv("CORECODER_ENFORCE_PLANNING", "1")
    msg = planning_guard("write_file")
    assert msg and "规划拦截" in msg


def test_guard_enforces_step_discipline():
    """Once a plan exists, a mutation must wait until the step is in_progress."""
    set_active_plan(TaskPlan(goal="g", items=[TodoItem(id="s1", description="d")]))
    assert planning_guard("write_file") is not None  # s1 still pending

    plan = __import__("corecoder.planner", fromlist=["get_active_plan"]).get_active_plan()
    plan.items[0].status = TodoStatus.IN_PROGRESS
    assert planning_guard("write_file") is None  # in_progress -> allowed
