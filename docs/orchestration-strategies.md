# 编排策略选择指南

`Orchestrator` 支持四种执行策略（`corecoder/agents/orchestrator.py`）。本文
帮你按任务形状选对策略。

## 策略一览

| 策略 | 语义 | 典型场景 |
|------|------|---------|
| `SEQUENTIAL` | A → B → C，后一个等前一个 | 有依赖链：探索 → 计划 → 实现 → 审查 |
| `PARALLEL` | A + B + C 同时跑 | 独立子任务：同时探索多个模块、多文件审查 |
| `CONDITIONAL` | 前一步成功才继续，失败即停 | 门禁式：实现失败就不再审查 |
| `AUTO` | 由 Orchestrator 选择 | 未显式指定的默认 |

## 决策树

```
任务可以拆成无依赖的独立子任务吗？
├─ 是 ─────────────────────────────► PARALLEL（加速比 > 1 才划算）
└─ 否 ── 子任务之间有先后依赖吗？
   ├─ 是 ── 前一步失败后面还有意义吗？
   │   ├─ 没有 ─────────────────────► CONDITIONAL（失败即停，省 Token）
   │   └─ 有 ───────────────────────► SEQUENTIAL
   └─ 否 ───────────────────────────► SEQUENTIAL（或先 AUTO 观察）
```

## 性能权衡

- **加速比** = 串行耗时 / 并行耗时。并行只有在子任务真正独立时才>1；若它们
  共享黑板上同一把锁、或互相等待，并行反而拖慢（评测指标 `speedup_ratio`
  会暴露这点，见 `corecoder/eval/metrics.py`）。
- **上下文膨胀率** = Subagent 摘要 Token / 原始对话 Token。并行跑得越多，
  汇总到主上下文的摘要也越多。子任务 ≤ 4 个时通常划算；超过 8 个时优先考虑
  把任务合并或分层。

## 示例

```python
from corecoder.agents import Blackboard, OrchestrationStrategy, Orchestrator

orch = Orchestrator(blackboard=Blackboard())

# 并行：独立探索两个模块
await orch.orchestrate(
    "调研 auth 与 billing 模块",
    strategy=OrchestrationStrategy.PARALLEL,
    subtasks=[
        {"subagent_name": "explorer", "task": "调研 auth"},
        {"subagent_name": "explorer", "task": "调研 billing"},
    ],
)

# 串行：探索 → 实现（有依赖）
await orch.orchestrate(
    "给登录加验证码",
    strategy=OrchestrationStrategy.SEQUENTIAL,
    subtasks=[
        {"subagent_name": "explorer", "task": "定位登录代码"},
        {"subagent_name": "implementer", "task": "按计划实现验证码"},
    ],
)
```

## 熔断提醒

同一 Subagent 连续失败 3 次后会被熔断（下次编排直接返回
`CIRCUIT_BREAKER_OPEN` 错误信封）。若某个角色频繁熔断，先查它的工具白名单
和模型档位，而不是加大重试次数。
