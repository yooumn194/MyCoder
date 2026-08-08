# Subagent 开发指南

本指南面向想给 CoreCoder 添加自定义 Subagent 的开发者。所有 Subagent 都通过
RFC v1.0.1 信封报告结果（`corecoder/contracts/envelope.py`），你只需要定义
「角色」——工具白名单、模型档位、预算——其余交给 `SubagentRunner` 与
`Orchestrator`。

## 1. 一个 Subagent 的完整定义

`SubagentDefinition` 只需要六类信息：

```python
from corecoder.agents import SubagentDefinition

code_tester = SubagentDefinition(
    name="code_tester",
    description="运行测试并报告通过/失败/跳过情况。返回 GeneralResult。",
    system_prompt=(
        "You are a test-runner subagent. Run the relevant tests and report "
        "pass/fail/skip counts as structured data."
    ),
    allowed_tools=["read_file", "execute_in_sandbox", "grep_search"],
    model_tier="standard",   # None = 继承主 Agent 的模型路由
    max_turns=12,
    max_tokens=60000,
    timeout_seconds=300,
    read_only=True,          # 只读 Subagent 的工具白名单里不应有写工具
    retry_on_failure=True,
)
```

## 2. 注册到目录

```python
from corecoder.agents.definition import BUILTIN_SUBAGENTS

BUILTIN_SUBAGENTS["code_tester"] = code_tester
```

或参考 `corecoder/agents/definition.py` 里内置的 explorer / planner /
implementer / reviewer。

## 3. 工具白名单：只读语义

`read_only=True` 是**约定**而非硬拦截：它由 `allowed_tools` 保证（只读角色
的白名单不含 `write_file` / `edit_file` / `execute_in_sandbox`）。添加新
Subagent 时，保持「只读角色永远不带写工具」这条纪律。

## 4. 结果信封（v1.0.1）

每个 Subagent 的最终输出必须是 RFC v1.0.1 信封（Pydantic 校验）：

```json
{
  "schema_version": "1.0.1",
  "meta": {
    "task_id": "...", "subagent_name": "code_tester",
    "subagent_instance_id": "...", "started_at": "...", "finished_at": "...", "duration_ms": 1200
  },
  "status": "success",
  "summary": "运行了 12 个测试：11 通过 1 失败",
  "confidence": "high",
  "result": { "type": "general", "output": "11/12 passed" },
  "artifacts": [{ "path": "test_report.txt", "action": "created" }]
}
```

关键规则：
- `status=success` 不能带 `error`；`status=failed` 必须带 `error` 且不含 `result`
- `status=partial` 必须同时带 `error` + `completeness_ratio`（严格 `0 < x < 1`）
- `summary` ≤ 500 字符，是 Orchestrator 唯一的决策文本
- `subagent_instance_id` 由 Orchestrator 注入，Subagent 无需自造

## 5. 内部工具调用也有契约

`corecoder/agents/tool_validator.py` 校验 Subagent 内部工具调用的返回值（例如
`grep_search` 必须返回 `matches` + `total_count`）。`SubagentRunner` 持有一个
`ToolOutputValidator`；直接调工具的 Harness 用它做 `validate_and_retry`
（最多重试 2 次，异常输出不会流进推理）。

## 6. 示例：端到端运行

```python
import asyncio
from corecoder.agents import Blackboard, Orchestrator, SubagentRunner
from corecoder.agents.definition import BUILTIN_SUBAGENTS

async def main():
    orch = Orchestrator(blackboard=Blackboard())
    env = await orch.spawn_subagent(
        "code_tester", "run pytest and summarize", parent_context={"task_id": "t1"},
    )
    print(env.status, env.summary)

asyncio.run(main())
```
