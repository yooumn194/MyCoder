"""todo_write / todo_update: structured plan management for the agent.

The write path validates the plan BEFORE storing it (optimization point #4):
a plan with a cycle or dangling dependency is rejected outright so a broken
plan never pollutes context; an over-granular plan is stored but the warning
is surfaced.
"""

from ..planner import (
    PlanStore,
    PlanValidator,
    TaskPlan,
    TodoItem,
    TodoStatus,
    clear_pending_memory_section,
    get_pending_memory_section,
    get_plan_store,
    notify_plan_complete,
    set_active_plan,
)
from .base import Tool


class TodoWriteTool(Tool):
    name = "todo_write"
    description = (
        "Create a structured execution plan. Pass task_goal and a list of "
        "steps, each with an id, a description, and optional depends_on ids. "
        "Invalid plans (cyclic or dangling dependencies) are rejected. After "
        "creating the plan, mark each step in_progress via todo_update before "
        "executing it."
    )
    parameters = {
        "type": "object",
        "properties": {
            "task_goal": {
                "type": "string",
                "description": "The overall goal this plan achieves",
            },
            "todos": {
                "type": "array",
                "description": "List of steps",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "description": {"type": "string"},
                        "depends_on": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["id", "description"],
                },
            },
        },
        "required": ["task_goal", "todos"],
    }

    def __init__(self, *, store: PlanStore | None = None) -> None:
        self._store = store or get_plan_store()

    def execute(self, task_goal: str, todos: list[dict]) -> str:
        items = [
            TodoItem(
                id=str(t.get("id", "")),
                description=str(t.get("description", "")),
                depends_on=[str(d) for d in t.get("depends_on", [])],
            )
            for t in todos
        ]

        # === 写入前校验 ===
        validation_error = PlanValidator.validate(items)
        if validation_error and validation_error.startswith("❌"):
            return validation_error  # 硬性拒绝，迫使 Agent 修正

        plan = TaskPlan(goal=task_goal, items=items)
        self._store.save(plan)
        set_active_plan(plan)

        result = f"✅ 计划已创建：{len(items)} 步"
        if validation_error:  # ⚠️ 警告信息附带返回
            result += f"\n{validation_error}"

        # Phase 5: prepend the memory section staged by planning_guard (if any)
        # so a fresh plan is informed by cross-session memories.
        section = get_pending_memory_section()
        clear_pending_memory_section()
        if section:
            result = section + result
        return result


class TodoUpdateTool(Tool):
    name = "todo_update"
    description = (
        "Update a step's status in the active plan (pending / in_progress / "
        "done / failed). Mark a step in_progress before executing it; mark it "
        "done when finished."
    )
    parameters = {
        "type": "object",
        "properties": {
            "step_id": {"type": "string", "description": "Step id from the plan"},
            "status": {
                "type": "string",
                "enum": ["pending", "in_progress", "done", "failed"],
                "description": "New status",
            },
        },
        "required": ["step_id", "status"],
    }

    def __init__(self, *, store: PlanStore | None = None) -> None:
        self._store = store or get_plan_store()

    def execute(self, step_id: str, status: str) -> str:
        from ..planner import get_active_plan

        plan = get_active_plan()
        if plan is None:
            return "❌ 无活跃计划，请先调用 todo_write"

        item = plan.item_map().get(step_id)
        if item is None:
            return f"❌ 步骤 [{step_id}] 不存在"

        valid_statuses = {
            TodoStatus.PENDING,
            TodoStatus.IN_PROGRESS,
            TodoStatus.DONE,
            TodoStatus.FAILED,
        }
        if status not in valid_statuses:
            return f"❌ 无效状态: {status}"

        item.status = status
        self._store.save(plan)
        set_active_plan(plan)
        # Phase 5: when the whole plan is done, notify the memory integration
        # (default hook is a no-op) so the finished plan is distilled.
        if status == TodoStatus.DONE and all(
            t.status == TodoStatus.DONE for t in plan.items
        ):
            notify_plan_complete(plan)
        return f"✅ 步骤 [{step_id}] 已标记为 {status}"
