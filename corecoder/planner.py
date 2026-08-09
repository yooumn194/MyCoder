"""Planning engine: TodoItem / TaskPlan / PlanValidator / PlanStore.

Phase 3's "smart decision-maker" upgrade. The agent writes structured plans
before mutating code, and a validation layer keeps bad plans from polluting
context. Plans persist to ~/.corecoder/plans/plan.json so a crashed run can be
resumed.

The planning_guard() in this module is the hard-interception layer that
`agent._exec_tool` consults before every tool call — see tools/planner_guard
comments for the soft/hard semantics.
"""

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

PLAN_DIR_ENV = "CORECODER_PLAN_DIR"
ENFORCE_PLANNING_ENV = "CORECODER_ENFORCE_PLANNING"
_TRUE = {"1", "true", "yes", "on"}


class TodoStatus:
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"


@dataclass
class TodoItem:
    id: str
    description: str
    status: str = TodoStatus.PENDING
    depends_on: list[str] = field(default_factory=list)


@dataclass
class TaskPlan:
    goal: str
    items: list[TodoItem]
    created_at: float = field(default_factory=time.time)

    def item_map(self) -> dict[str, TodoItem]:
        return {t.id: t for t in self.items}

    def get_current_step(self) -> TodoItem | None:
        """The first PENDING step whose dependencies are all done.

        This is what the planning guard uses: a mutation outside the current
        step violates plan discipline.
        """
        done = {t.id for t in self.items if t.status == TodoStatus.DONE}
        for t in self.items:
            if t.status == TodoStatus.PENDING and set(t.depends_on) <= done:
                return t
        return None


class PlanValidator:
    """Lightweight guard against invalid plans — reject before polluting context."""

    @staticmethod
    def validate(todos: list[TodoItem]) -> str | None:
        """Return a rejection (❌) or warning (⚠️) message, or None if valid.

        A `❌` message is a hard rejection the tool refuses to write; a `⚠️`
        message is a warning that is written but surfaced to the agent.
        """
        if PlanValidator._has_cycle(todos):
            return "❌ 计划存在循环依赖，请检查 depends_on 字段"

        all_ids = {t.id for t in todos}
        for t in todos:
            missing = set(t.depends_on) - all_ids
            if missing:
                return f"❌ 步骤 [{t.id}] 依赖了不存在的步骤: {missing}"

        if len(todos) > 10:
            return "⚠️ 计划超过10步，建议拆分为子任务或确认是否过于细碎"

        return None

    @staticmethod
    def _has_cycle(todos: list[TodoItem]) -> bool:
        """DFS cycle detection over depends_on edges."""
        id_map = {t.id: t for t in todos}
        visiting: set[str] = set()
        visited: set[str] = set()

        def dfs(tid: str) -> bool:
            if tid in visiting:
                return True
            if tid in visited:
                return False
            visiting.add(tid)
            for dep in id_map[tid].depends_on:
                if dep not in id_map:
                    continue  # dangling deps are reported separately
                if dfs(dep):
                    return True
            visiting.discard(tid)
            visited.add(tid)
            return False

        return any(dfs(t.id) for t in todos)


class PlanStore:
    """File-backed persistence for the active plan."""

    def __init__(self, base_dir: Path | str | None = None) -> None:
        base = Path(base_dir or os.getenv(PLAN_DIR_ENV, "~/.corecoder/plans"))
        base = base.expanduser().resolve()
        base.mkdir(parents=True, exist_ok=True)
        self.path = base / "plan.json"

    def save(self, plan: TaskPlan) -> None:
        self.path.write_text(
            json.dumps(self._to_dict(plan), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self) -> TaskPlan | None:
        if not self.path.exists():
            return None
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        return self._from_dict(data)

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)

    # -------------------------------------------------------------- serialization

    @staticmethod
    def _to_dict(plan: TaskPlan) -> dict:
        return {
            "goal": plan.goal,
            "created_at": plan.created_at,
            "items": [
                {
                    "id": t.id,
                    "description": t.description,
                    "status": t.status,
                    "depends_on": list(t.depends_on),
                }
                for t in plan.items
            ],
        }

    @staticmethod
    def _from_dict(data: dict) -> TaskPlan:
        return TaskPlan(
            goal=data.get("goal", ""),
            created_at=data.get("created_at", 0.0),
            items=[TodoItem(**t) for t in data.get("items", [])],
        )


# ---------------------------------------------------------------------------
# active-plan accessors (in-memory, process-scoped) + the planning guard
# ---------------------------------------------------------------------------

_active_store: PlanStore | None = None
_active_plan: TaskPlan | None = None

EXPLORATION_TOOLS = {"grep_search", "list_files", "read_file", "glob"}
PLANNING_TOOLS = {"todo_write", "todo_update"}
MUTATION_TOOLS = {"write_file", "edit_file", "execute_in_sandbox", "sync_workspace"}


def get_plan_store() -> PlanStore:
    global _active_store
    if _active_store is None:
        _active_store = PlanStore()
    return _active_store


def get_active_plan() -> TaskPlan | None:
    """The in-memory active plan, else the persisted one (crash recovery)."""
    global _active_plan
    if _active_plan is None:
        _active_plan = get_plan_store().load()
    return _active_plan


def has_active_plan() -> bool:
    return get_active_plan() is not None


def set_active_plan(plan: TaskPlan | None) -> None:
    global _active_plan
    _active_plan = plan


def clear_active_plan() -> None:
    global _active_plan
    _active_plan = None
    get_plan_store().clear()


def enforcement_enabled() -> bool:
    return os.getenv(ENFORCE_PLANNING_ENV, "").strip().lower() in _TRUE


# ---------------------------------------------------------------------------
# Phase 5 memory hooks (default no-ops; wired by memory/integration.py).
# Keeping them here (not importing memory) avoids an import cycle.
# ---------------------------------------------------------------------------
_memory_injector = None  # fn(query: str) -> str  (a memory prompt section)
_pending_memory_section = ""
_plan_complete_hook = None  # fn(plan: TaskPlan) -> None


def set_memory_injector(fn) -> None:
    global _memory_injector
    _memory_injector = fn


def get_pending_memory_section() -> str:
    """The memory section staged by planning_guard for the next todo_write."""
    return _pending_memory_section


def clear_pending_memory_section() -> None:
    global _pending_memory_section
    _pending_memory_section = ""


def set_plan_complete_hook(fn) -> None:
    global _plan_complete_hook
    _plan_complete_hook = fn


def notify_plan_complete(plan) -> None:
    """Called by todo_update when the last step becomes done."""
    if _plan_complete_hook is not None:
        try:
            _plan_complete_hook(plan)
        except Exception:  # noqa: BLE001 - settlement must never break the tool
            pass


def planning_guard(tool_name: str, query: str | None = None) -> str | None:
    """Consulted by agent._exec_tool before every tool call.

    Returns an error string to BLOCK the tool, or None to allow it.

    Soft/hard semantics (optimization point #1):
      * exploration + planning tools are always allowed;
      * with NO active plan: allowed by default (preserves Phase 1/2 open-ended
        behaviour), hard-blocked only when CORECODER_ENFORCE_PLANNING=1;
      * with an ACTIVE plan: plan discipline is enforced — the current step
        must be marked in_progress before a mutation tool runs.

    Phase 5 (side effect, never blocks): when the agent is about to create a
    plan (todo_write) and a query is supplied, a relevant-memory section is
    staged for TodoWriteTool to prepend to its output. `planning_guard("todo_write")`
    still returns None regardless — the memory section is a side channel.
    """
    global _pending_memory_section
    if tool_name == "todo_write" and query and _memory_injector is not None:
        try:
            _pending_memory_section = _memory_injector(query) or ""
        except Exception:  # noqa: BLE001 - injection must never break planning
            _pending_memory_section = ""
    if tool_name in EXPLORATION_TOOLS | PLANNING_TOOLS:
        return None
    if tool_name not in MUTATION_TOOLS and not tool_name.startswith("mcp_"):
        # mcp_* tools (Phase 3.5 adapters) are treated as ordinary mutation
        # tools, so they obey the same plan discipline.
        return None

    plan = get_active_plan()
    if plan is None:
        # 设计决策：无计划时默认放行，避免破坏 Phase 1/2 的开放式交互探索。
        # 切换条件：当用户/系统设置 CORECODER_ENFORCE_PLANNING=1 时，无计划 +
        # 非探索工具 → 硬拦截。未来 Phase 4 多 Agent 编排时，Orchestrator 可
        # 强制开启此开关。这不是"忘了实现拦截"，而是有意的 soft-default。
        if enforcement_enabled():
            return (
                "[规划拦截] 当前无活跃计划，禁止直接执行修改类工具。\n"
                "请先调用 todo_write 创建执行计划，或使用 grep_search/read_file "
                "等探索工具了解代码库。"
            )
        return None  # soft mode: no plan -> open-ended execution (Phase 1/2)

    current = plan.get_current_step()
    if current and current.status != TodoStatus.IN_PROGRESS:
        return (
            f"[规划拦截] 当前步骤 [{current.id}] 未标记为 in_progress，"
            f"请先 todo_update 将其置为 in_progress。"
        )
    return None
