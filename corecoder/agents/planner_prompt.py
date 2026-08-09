"""Prompt templates for the TaskPlanner — kept separate so they can be tuned
without touching planner.py.

The subagent catalog mirrors `definition.py`'s BUILTIN_SUBAGENTS: the planner
may only choose from those roles, and each role's capability description here
is what guides the model's assignment decisions.

Security note: user text (the task) is interpolated with str.replace-style
concatenation, never str.format, so a task containing braces (e.g. a code
snippet like `if x {`) cannot break the template or inject placeholders.
"""

# The subagent catalog the planner may choose from (must match
# agents/definition.py BUILTIN_SUBAGENTS).
_SUBAGENT_CATALOG = """\
可用 subagent 类型（只能从以下四种里选）：
- explorer:      只读代码搜索与定位（grep/list_files/read_file），返回发现结果，不改代码
- planner:       设计实现计划（可用 todo_write），不写代码，只输出方案
- implementer:   执行代码修改并运行测试验证（write_file/execute_in_sandbox）
- reviewer:      只读代码审查，检查正确性、质量与安全问题，不改代码"""

# Only {subagent_catalog} and {max_subtasks} are real placeholders. The
# `{{"id": ...}}` line is escaped so .format unescapes it into a literal JSON
# element example.
_SYSTEM_HEADER = """\
你是任务规划器：把用户的复杂编程任务拆解成一组可并行的子任务 DAG，交给不同类型的 subagent 执行。

{subagent_catalog}

输出约束：
1. 严格输出一个 JSON 数组，不要 markdown 代码块、不要解释、不要额外文字
2. 数组元素格式：{{"id": "t1", "subagent_name": "...", "instruction": "...", "depends_on": ["..."], "estimated_tokens": 1000}}
3. 子任务数 <= {max_subtasks}；instruction 必须自包含（子 agent 看不到其它任务内容）
4. depends_on 只能引用本数组里已存在的 id；无依赖的任务留空数组
5. 任务之间真有数据/顺序依赖才连边：能并行的就并行，避免不必要的串行
6. estimated_tokens 是你对该子任务耗 token 量的预估（用于预算分配）
"""

# Plain static text with real single braces — concatenated AFTER .format so it
# is never re-processed as a format string.
_EXAMPLES = """\
示例 1（简单 bugfix → 单任务）：
任务：修复 src/parse.py 中 parse_line 在空输入时抛 IndexError 的 bug
输出：[{"id": "t1", "subagent_name": "implementer", "instruction": "修复 parse_line 对空输入的 IndexError，补一条空输入用例并运行 pytest", "depends_on": [], "estimated_tokens": 2000}]

示例 2（跨文件重构 → 3 任务 DAG）：
任务：把认证逻辑从 controller 里抽到独立 service 模块，并更新调用方
输出：[{"id": "t1", "subagent_name": "explorer", "instruction": "定位所有调用认证逻辑的 controller 文件与依赖关系", "depends_on": [], "estimated_tokens": 1200},
{"id": "t2", "subagent_name": "implementer", "instruction": "新建 auth service 模块，把 controller 里的认证逻辑迁移过去", "depends_on": ["t1"], "estimated_tokens": 4000},
{"id": "t3", "subagent_name": "reviewer", "instruction": "审查 auth service 迁移：确认无遗漏调用方、无安全回归", "depends_on": ["t2"], "estimated_tokens": 1500}]
"""


def build_system_prompt(max_subtasks: int) -> str:
    """Render the system prompt with the configured subtask cap."""
    return _SYSTEM_HEADER.format(
        subagent_catalog=_SUBAGENT_CATALOG, max_subtasks=max_subtasks
    ) + _EXAMPLES


def build_user_prompt(task: str, context: dict | None) -> str:
    """Render the user turn by concatenation (never .format) so braces inside
    the user's task cannot break the template or inject placeholders."""
    return (
        "任务：\n"
        + str(task)
        + "\n\n附加上下文（可为空）：\n"
        + str(context or {})
    )
