# MyCoder

一个从零实现、零框架依赖的 AI 编程 Agent。核心就是一个 while 循环：让模型循环"思考 → 调工具 → 看结果"，直到任务完成。全部实现都在这个仓库里——从 agent 主循环到 Docker 沙箱、从混合检索记忆到多 Agent 编排、从三层安全护栏到完整的性能可观测。

**特性一览**

- **自研 Agent 主循环**：不依赖 LangChain / CrewAI，工具循环、规划、纠错、防死循环全部手写
- **Docker 沙箱隔离**：命令在只读根文件系统、无网络、丢弃全部 Linux 权限的容器里执行
- **混合检索记忆**：BM25（jieba 分词）+ 语义向量（BGE）经 RRF 融合，跨会话持久
- **多 Agent 编排**：子代理委派、RFC 信封、熔断自愈、动态重规划
- **三层安全护栏**：输入注入检测、工具白名单与确认、输出密钥脱敏
- **工程化完备**：token 预算、限流、SLO 告警、性能指标（TTFT / 延迟 / 成本）、监控报告、评测与压测

---

## 快速开始

```bash
# 安装（Python 3.11+）
pip install -e .

# 配置（任选一种 LLM）
export MYCODER_API_KEY=sk-...            # 或 OPENAI_API_KEY
export OPENAI_BASE_URL=https://api.deepseek.com   # OpenAI 兼容即可

# 构建沙箱镜像（一次）
docker build -t mycoder-sandbox:3.12 -f sandbox/Dockerfile sandbox/

# 交互式使用
mycoder

# 一次性任务
mycoder -p "给 parse_config() 加错误处理"

# 记忆语义向量（可选，装了才启用 BGE）
pip install fastembed sqlite-vec   # 或 pip install -e '.[memory-embed,memory-vec]'
```

---

## 一次对话，内部发生了什么

这是按**真实运行流程**（从你输入到拿到回答）的路线图，每步标注所用技术栈与实现方式。

```
你输入「帮我重构这个模块」
        │
        ▼
① Agent 主循环（mycoder/agent.py）
        │  注入防御 → 上下文压缩 → LLM 携带工具集 → 执行工具 → 回灌结果 → 循环
        ▼
② LLM 推理（mycoder/llm.py）
        │  OpenAI 兼容 API · 流式 · 工具调用解析 · 自动重试
        ▼
③ 工具调用（mycoder/tools/）
        │  26 个工具：沙箱执行 / 文件读写 / 搜索 / 记忆 / 子代理 / 规划
        ▼
④ Docker 沙箱（mycoder/sandbox/）
        │  命令在隔离容器执行，/workspace 增量同步回宿主
        ▼
⑤ 记忆系统（mycoder/memory/）
        │  跨会话事实 / 决策 / 经验，混合检索召回
        ▼
⑥ 多 Agent（mycoder/agents/）
        │  复杂任务派子代理，信封收结果，偏差自动重规划
        ▼
⑦ 安全护栏（mycoder/tools/security.py 等）
        │  输入 / 执行 / 输出三层拦截 + 密钥脱敏
        ▼
回复你，并记录全部 trace 到可观测层
```

### ① Agent 主循环

`Agent.chat()` 是全部行为的核心，约 40 行：

```python
for _ in range(max_rounds):          # 防死循环：轮次上限
    resp = llm.chat(messages, tools) # 携带工具 schema 的 LLM 调用
    if not resp.tool_calls:
        return resp.content          # 模型决定回答，结束
    for tc in resp.tool_calls:
        result = exec_tool(tc)       # 执行工具
        messages.append(tool 结果)   # 观察回灌，进入下一轮
```

- **注入防御**：用户输入先过正则 + LLM 分类器，工具结果包裹 `<tool_output>` 标签并声明"是数据不是指令"
- **上下文压缩**：超 50% 截断工具输出、超 70% LLM 摘要旧轮、超 90% 硬折叠——压缩掉的内容经回调沉降进长期记忆库
- **推理策略**：ReAct（默认）/ Plan-and-Execute / Reflection 三选一，未指定时**按任务自动路由**（重构→plan_execute、修 bug→reflection），`/strategy` 可运行时切换
- **工具选择**：按当前会话相关性注入 Top-K 工具（核心 11 个常驻 + 相关度排序），省 token 且减少误选
- **幂等与纠错**：相同 `(工具, 参数)` 幂等调用命中缓存不重复执行；失败按分类确定性重试（可重试 2 次、超时翻倍），非幂等写不自动重试

### ② LLM 推理层

- 对接任意 OpenAI 兼容接口（`LLM` 类），也支持 LiteLLM 走 100+ 提供商
- **流式输出**，并测量 TTFT（首 token 延迟）；token 用量精确统计（tiktoken 兜底估算）
- 模型分级路由：简单子任务走 fast 档、复杂走 powerful 档（`config/model_routing.yaml` 规则）

### ③ 工具层

| 工具 | 用途 | 实现要点 |
|---|---|---|
| `execute_in_sandbox` | 沙箱里跑 shell | Docker 硬化容器，见 §④ |
| `sync_workspace` | 拉回沙箱变更 | `docker diff` 增量，`git status` 感知改动 |
| `read_file` / `write_file` / `edit_file` | 文件读写 | 路径守卫防越权，`/workspace` 自动映射 |
| `grep_search` / `list_files` | 代码搜索 | rg 优先 + 纯 Python 兜底，路径受控 |
| `memory_save` / `search` / … | 跨会话记忆 | 混合检索 + 去重 + 衰减 |
| `spawn_subagent` | 派子代理 | 编排器委派，见 §⑥ |
| `todo_write` / `todo_update` | 规划执行 | 计划纪律（见安全 §⑦） |

### ④ Docker 沙箱（核心安全屏障）

命令在**一次性硬化容器**里执行，不是正则黑名单：

```
read_only 只读根文件系统 · tmpfs /tmp 64m · network none 无网络
user=sandbox 非 root · no-new-privileges · cap_drop=ALL 零权限
mem 512m · cpu 0.5 核 · pids 128（防 fork bomb）
```

- 项目目录只读挂载为 `/src`，可写的是独立 `/workspace` 卷——被攻破也只能读项目、改不了、出不去
- 超时命令杀容器自愈重建，OOM 熔断（连挂 2 次停止重试）
- **空闲自动回收**：闲置 `MYCODER_SANDBOX_IDLE_TIMEOUT`（默认 10 分钟）自动停容器、保留卷，下次调用无缝重启
- **退出清理**：进程退出（正常 / Ctrl+C / kill）经 `atexit` 钩子关沙箱容器、MCP 子进程、记忆库连接

### ⑤ 记忆系统

跨会话的长期记忆，两个维度：

- **存储**：SQLite（全文索引 FTS5 + 语义向量表）+ 可选 BGE 向量（`fastembed` + `sqlite-vec`）
- **检索**：混合检索——BM25（jieba 分词）与语义向量分别召回，**RRF 融合**（k=60）排序，再经规则重排
- **生命周期**：写入去重（余弦 0.85）、置信度衰减（30 天）、错误记忆审计与纠正、低价值记忆滚动合并
- **闭环**：对话被压缩时，关键事实经 `on_compressed` 回调**沉降进记忆库**——上下文压缩不丢信息
- **Agentic RAG**：检索→判断够不够→不足改写再查，最多 3 轮
- **经验沉淀**：多 Agent 重规划的"偏差→策略→是否恢复"记录也写进记忆，下次可检索复用

### ⑥ 多 Agent 编排

复杂任务经 `Orchestrator` 拆给子代理（explorer / planner / implementer / reviewer）：

- **RFC 信封**：每个子代理返回 Pydantic 强校验的 v1.0.1 结果信封（状态/错误/制品），杜绝脏数据
- **模型分级路由**：explorer 用 fast 模型、planner 用 powerful——按角色省钱
- **熔断自愈**：某子代理连败 3 次熔断，冷却后半开探针自动恢复
- **动态重规划**：执行中检测三类偏差——硬失败（重试 1 次 / 永久跳过）、软偏差（插入修正节点）、目标漂移（重新拆解剩余任务），上限 3 次防死循环

### ⑦ 安全护栏（三层）

| 层 | 位置 | 手段 |
|---|---|---|
| 输入层 | `Agent.chat` | 正则 + LLM 分类器检测注入；工具结果 `<tool_output>` 角色隔离 |
| 执行层 | 工具边界 | 路径守卫、危险命令预检、高风险操作确认（ConfirmPolicy）、计划纪律、MCP 白名单 + 参数正则 |
| 输出层 | `redact_output` | 最终回复脱敏（API key / Bearer / PEM）；检索回答带编号引用可溯源 |

### ⑧ 可观测（每次运行都被记录）

- **LLM trace**：每次调用记延迟 / token / TTFT / 成本 / 错误，按会话聚合 avg/p95
- **工具指标**：成功率 / 重试率 / 工具失败率 / 执行时长
- **预算与限流**：session token 预算（默认 100k），API 限流（令牌桶/漏桶/滑动窗口）
- **SLO 告警**：成功率 <90%、p95 >5s 触发告警（防抖）
- **监控报告**：`GET /v1/agent/report` 一页聚合所有指标 + 生产任务成功率

---

## 技术栈

| 层 | 技术 |
|---|---|
| 语言 | Python 3.11+，零框架依赖（自研 agent loop） |
| LLM | OpenAI 兼容 API / LiteLLM，流式 |
| 沙箱 | Docker SDK，容器硬化参数 |
| 记忆 | SQLite + FTS5 + 语义向量（BGE，可选）+ jieba 分词 |
| 数据模型 | Pydantic（信封/参数校验） |
| 日志/追踪 | structlog（结构化）+ 自研 LLMTracer |
| 服务层 | FastAPI + Redis/SQLite 状态后端 |
| 评测 | pytest / httpx / LLM-as-Judge / RAG 指标 / 并发压测 |

---

## 配置（环境变量）

| 变量 | 默认 | 说明 |
|---|---|---|
| `MYCODER_MODEL` | `gpt-5.5` | 模型名 |
| `MYCODER_API_KEY` / `OPENAI_API_KEY` | — | API 密钥 |
| `OPENAI_BASE_URL` | — | OpenAI 兼容端点 |
| `MYCODER_MAX_CONTEXT` | `128000` | 上下文窗口 |
| `MYCODER_SANDBOX_MEM/CPU/PIDS` | `512m/0.5/128` | 沙箱资源 |
| `MYCODER_SANDBOX_IDLE_TIMEOUT` | `600` | 沙箱空闲回收（秒，0 禁用） |
| `MYCODER_SESSION_BUDGET` | `100000` | 会话 token 预算 |
| `MYCODER_RATE_LIMIT` | 关 | API 限流（次/分） |
| `MYCODER_INJECTION_GUARD` | `on` | 注入防御开关 |
| `MYCODER_MODEL_TIER` | `standard` | 模型分级 |

---

## 服务层（可选）

`api/` 提供 FastAPI 服务：`POST /v1/agent/run` 后台跑任务、`/status` 查状态、`/cost` 查成本、`/metrics` 看成功率、`/report` 看监控快照。

```bash
uvicorn api.server:app    # 或 docker compose up --build
```

---

## 评测与压测

```bash
# 代码任务 Pass@1（黑盒 HTTP，需服务在跑）
python -m eval_bench.runner && python -m eval_bench.scorer --results results/<run>

# RAG 检索指标（真实文档 + 金标准查询，离线）
python -m eval_bench.rag_eval --doc README.md --queries eval_bench/rag_queries.json --compare --embedder config

# LLM-as-Judge 质量打分
python -m eval_bench.judge_run

# 并发压测（QPS / p95）
python -m eval_bench.loadtest --concurrency 4 --requests 20
```

---

## 致谢

本项目由 [CoreCoder](https://github.com/he-yufeng/CoreCoder) 更名、二次开发而来。感谢原作者 he-yufeng 的开源贡献——正是它的设计给了 MyCoder 生长的土壤。

## License

MIT
