"""End-to-end Phase 3 tests: search -> plan -> modify -> verify, guard enforced.

These drive tools through the real agent dispatch (`Agent._exec_tool`), so the
planning guard and the self-correction wrapper are exercised exactly as they
run in production.
"""

from types import SimpleNamespace

import pytest

from corecoder import planner as planner_mod
from corecoder.agent import Agent
from corecoder.planner import PlanStore
from corecoder.tools.grep_search import GrepSearchTool
from corecoder.tools.read_file import ReadFileTool
from corecoder.tools.todo_tools import TodoUpdateTool, TodoWriteTool
from corecoder.tools.write import WriteFileTool


def _tc(name, **args):
    return SimpleNamespace(name=name, arguments=args)


class _FakeLLM:
    """Not used — these tests drive _exec_tool directly."""


@pytest.fixture(autouse=True)
def _clean_planner(monkeypatch, tmp_path):
    store = PlanStore(base_dir=tmp_path / "plans")
    monkeypatch.setattr(planner_mod, "_active_store", store)
    planner_mod.clear_active_plan()
    yield
    planner_mod.clear_active_plan()


def _make_proj(tmp_path):
    (tmp_path / "app.py").write_text(
        "def add(a, b):\n    return a + b\n\n"
        "def sub(a, b):\n    return a - b\n",
        encoding="utf-8",
    )
    return tmp_path


def _agent(tmp_path):
    write = WriteFileTool()
    read = ReadFileTool(project_root=tmp_path)
    grep = GrepSearchTool(project_root=tmp_path, rg_path=None)
    todo_write = TodoWriteTool()
    todo_update = TodoUpdateTool()
    return Agent(llm=_FakeLLM(), tools=[write, read, grep, todo_write, todo_update])


def test_search_plan_modify_verify(tmp_path, monkeypatch):
    """The full Phase 3 workflow, with plan discipline enforced."""
    _make_proj(tmp_path)
    monkeypatch.setenv("CORECODER_ENFORCE_PLANNING", "1")
    agent = _agent(tmp_path)

    # 1. search to locate the function
    r = agent._exec_tool(_tc("grep_search", pattern="def add", file_types="py"))
    assert "app.py" in r

    # 2. plan
    out = agent._exec_tool(
        _tc("todo_write", task_goal="add subtract", todos=[
            {"id": "s1", "description": "read app.py"},
            {"id": "s2", "description": "add subtract function", "depends_on": ["s1"]},
        ])
    )
    assert "✅ 计划已创建" in out

    # 3. exploration stays allowed under enforcement
    assert "def add" in agent._exec_tool(_tc("read_file", file_path="app.py"))

    # 4. mutation BEFORE the current step is in_progress -> blocked by the guard
    blocked = agent._exec_tool(_tc("write_file", file_path=str(tmp_path / "app.py"), content="x"))
    assert "规划拦截" in blocked

    # 5. mark the step in_progress -> mutation now allowed
    agent._exec_tool(_tc("todo_update", step_id="s1", status="in_progress"))
    out = agent._exec_tool(
        _tc("write_file", file_path=str(tmp_path / "app.py"), content="def add(a,b):\n    return a+b\n")
    )
    assert "Wrote" in out

    # 6. verify via search
    r2 = agent._exec_tool(_tc("grep_search", pattern="def add", file_types="py"))
    assert "app.py" in r2


def test_open_ended_mode_preserves_phase12(monkeypatch, tmp_path):
    """No plan + enforcement off -> mutation runs (Phase 1/2 behaviour intact)."""
    _make_proj(tmp_path)
    monkeypatch.delenv("CORECODER_ENFORCE_PLANNING", raising=False)
    agent = _agent(tmp_path)

    out = agent._exec_tool(_tc("write_file", file_path=str(tmp_path / "x.py"), content="print(1)\n"))
    assert "Wrote" in out  # not intercepted


def test_transient_error_is_auto_retried_by_correction_loop(tmp_path, monkeypatch):
    """A tool that raises a transient error gets retried once, then succeeds."""
    _make_proj(tmp_path)

    calls = {"n": 0}

    class _FlakyWrite:
        name = "write_file"
        description = "flaky"
        parameters = {}

        def execute(self, file_path, content):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ConnectionResetError("transient")
            p = __import__("pathlib").Path(file_path)
            p.write_text(content, encoding="utf-8")
            return "Wrote 1 lines"

    agent = Agent(llm=_FakeLLM(), tools=[_FlakyWrite()])
    out = agent._exec_tool(_tc("write_file", file_path=str(tmp_path / "y.py"), content="z"))
    assert "Wrote" in out
    assert calls["n"] == 2  # 1 failed attempt + 1 retry
