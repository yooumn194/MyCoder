# MyCoder

一个基于 [CoreCoder](https://github.com/he-yufeng/CoreCoder) 深度二次开发、不依赖 Agent 框架的 AI 编程 Agent。核心是让模型循环“思考 → 调工具 → 看结果”，并在此基础上补齐沙箱、记忆、多 Agent 编排、服务可靠性与性能可观测。

**特性一览**

- **框架无关 Agent 主循环**：不依赖 LangChain / CrewAI，工具循环、规划、纠错、防死循环均在仓库内实现
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
# 包含 API 服务依赖的安装方式
pip install -e ".[api]"
mycoder-api --help


# 配置（任选一种 LLM）
export OPENAI_API_KEY=sk-...

# 或 OpenRouter（默认 minimax/minimax-m3:free）
export MYCODER_PROVIDER=openrouter
export OPENROUTER_API_KEY=sk-or-...

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
        │  21 个内置工具：沙箱执行 / 文件读写 / 搜索 / 记忆 / 子代理 / 规划
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

`Agent.chat()` 是全部行为的核心；抽去护栏、指标和纠错后，概念循环如下：

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
- **流式输出**，并测量 TTFT（首 token 延迟）；token 用量精确统计，单独记录 reasoning token（tiktoken 兜底估算）
- provider-aware 模型分级路由：简单子任务走 fast 档、复杂走 powerful 档；限流、超时、模型不可用时只在同一 provider 内安全降级（`config/model_routing.yaml`）

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
| `MYCODER_PROVIDER` | `openai` | `openai` / `deepseek` / `openrouter` / `ollama` / `litellm`；也会按专属密钥自动识别 |
| `MYCODER_PROFILE` | — | 快速完整切换 provider；忽略遗留的通用 model/base URL，使用该 provider 默认配置 |
| `MYCODER_MODEL` | provider 默认值 | 模型名；OpenRouter 默认 `minimax/minimax-m3:free` |
| `MYCODER_<PROVIDER>_MODEL` | provider 默认值 | provider 专属模型覆盖，如 `MYCODER_OPENROUTER_MODEL`，只需配置一次 |
| `MYCODER_<PROVIDER>_BASE_URL` | provider 默认值 | profile 模式下的 provider 专属端点覆盖 |
| `MYCODER_API_KEY` | — | 通用显式密钥覆盖 |
| `OPENAI_API_KEY` / `DEEPSEEK_API_KEY` / `OPENROUTER_API_KEY` | — | provider 专属 API 密钥 |
| `MYCODER_BASE_URL` / `OPENAI_BASE_URL` | provider 默认值 | OpenAI 兼容端点；OpenRouter 自动使用 `https://openrouter.ai/api/v1` |
| `OPENROUTER_SITE_URL` / `OPENROUTER_APP_NAME` | — | 可选的 OpenRouter 应用归属请求头 |
| `MYCODER_CHECKPOINT_DIR` | `~/.mycoder/checkpoints` | API 编排计划与已完成步骤的 checkpoint 目录 |
| `STATE_BACKEND` / `REDIS_URL` | `local` / `redis://localhost:6379/0` | `redis` 模式共享 session、job、lease、checkpoint、Trace、限流窗口和告警 |
| `MYCODER_OBSERVABILITY_PATH` | `.mycoder/api_state.db` | local 模式 Trace/限流/告警 SQLite 路径；默认与 API 状态共库 |
| `MYCODER_OBSERVABILITY_TTL_SECONDS` | `604800` | SQLite/Redis Trace、告警历史保留时间（秒） |
| `MYCODER_OBSERVABILITY_MAX_ALERTS` | `10000` | SQLite/Redis 告警历史最大条数 |
| `MYCODER_API_KEYS` | — | 租户 API key：JSON（`{"team":"secret"}`）或 `team=secret`；配置后 API 自动要求认证 |
| `MYCODER_REQUIRE_AUTH` | `false` | 设为 `true` 可禁止无 key 的本地开发模式 |
| `MYCODER_WORKSPACE_ROOT` | 服务 cwd | 认证租户工作区根；实际目录为 `<root>/<tenant>/<workspace_id>` |
| `MYCODER_JOB_LEASE_SECONDS` | `60` | worker job/session lease，运行期间自动续租 |
| `MYCODER_JOB_MAX_ATTEMPTS` | `3` | durable job 最大执行次数；超限后进入死信区 |
| `MYCODER_JOB_RETRY_BASE_SECONDS` / `MYCODER_JOB_RETRY_MAX_SECONDS` | `1` / `30` | 带抖动指数退避的起始/上限秒数 |
| `MYCODER_SSE_POLL_SECONDS` / `MYCODER_SSE_HEARTBEAT_SECONDS` | `0.25` / `15` | 结构化 SSE 状态采样与心跳间隔 |
| `MYCODER_MAX_CONTEXT` | `128000` | 上下文窗口 |
| `MYCODER_SANDBOX_MEM/CPU/PIDS` | `512m/0.5/128` | 沙箱资源 |
| `MYCODER_SANDBOX_IDLE_TIMEOUT` | `600` | 沙箱空闲回收（秒，0 禁用） |
| `MYCODER_SESSION_BUDGET` | `100000` | 会话 token 预算 |
| `MYCODER_RATE_LIMIT` | 关 | API 每 tenant+client 的请求/分钟；SQLite/Redis 跨 worker 原子共享 |
| `MYCODER_INJECTION_GUARD` | `on` | 注入防御开关 |
| `MYCODER_MODEL_TIER` | `standard` | 模型分级 |

---

## 服务层（可选）

`api/` 提供 FastAPI 服务：`POST /v1/agent/run` 先持久化 job，再由带 lease 的 worker 执行；worker 被杀后任务会从 checkpoint 恢复，连续失败则按指数退避重试并最终进入 dead letter。`/status`、`/events`（SSE）、`/dead-letter`、`/alerts`、`/cost`、`/metrics`、`/report` 都按租户隔离；Trace、成本来源、限流窗口、告警 cooldown/历史在 SQLite 或 Redis 中跨 worker 共享。

快速切换 provider 时不用修改 `.env` 中的 `MYCODER_MODEL`：

```bash
mycoder --provider openrouter
MYCODER_PROFILE=deepseek uvicorn api.server:app
```

`MYCODER_PROFILE`/`--provider` 会整体选择 `_PROVIDER_DEFAULTS` 中的模型、端点和专属 Key。若某个 provider 需要固定非默认模型，只需一次性设置，例如 `MYCODER_OPENROUTER_MODEL=openai/gpt-oss-120b:free`。

```bash
mycoder-api --host 0.0.0.0 --port 8000    # 或 docker compose up --build

# 多 worker/生产模式（先安装 pip install -e '.[api]'）
export STATE_BACKEND=redis REDIS_URL=redis://localhost:6379/0
export MYCODER_API_KEYS='{"team-a":"replace-with-a-secret"}'
mkdir -p team-a/repo-1  # 或把 MYCODER_WORKSPACE_ROOT 指向已有租户工作区根

# 首次运行；每个成功/部分成功的子步骤会写 checkpoint
curl -X POST http://localhost:8000/v1/agent/run \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: replace-with-a-secret' \
  -d '{"task":"实现并验证功能","session_id":"job-1","workspace_id":"repo-1"}'

# 失败或服务重启后，用同一 session_id + task 恢复；已完成步骤不会重跑
curl -X POST http://localhost:8000/v1/agent/run \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: replace-with-a-secret' \
  -d '{"task":"实现并验证功能","session_id":"job-1","workspace_id":"repo-1","resume":true}'

# status.checkpoint 返回 completed_steps / remaining_steps
curl -H 'X-API-Key: replace-with-a-secret' \
  http://localhost:8000/v1/agent/status/job-1

# SSE：progress / completed / failed 结构化事件；跨 worker 读取同一持久化状态
curl -N -H 'X-API-Key: replace-with-a-secret' \
  http://localhost:8000/v1/agent/events/job-1

# 运维查看本租户死信任务
curl -H 'X-API-Key: replace-with-a-secret' \
  http://localhost:8000/v1/agent/dead-letter
```

恢复请求会校验原 task，checkpoint 不存在返回 404、task 不一致或同 session 正在执行返回 409。任务成功后 checkpoint 自动清理；失败或进程中断时保留。

---

## 评测与压测

```bash
# 代码任务 Pass@1（黑盒 HTTP，需服务在跑）
python -m eval_bench.runner && python -m eval_bench.scorer --results results/<run>

# 固定 30 题 × 3 次 × 4 变体消融；输出 manifest、均值/标准差、失败分布
python -m eval_bench.matrix --dry-run
python -m eval_bench.matrix --base-url http://localhost:8000

# RAG 检索指标（真实文档 + 金标准查询，离线）
python -m eval_bench.rag_eval --doc README.md --queries eval_bench/rag_queries.json --compare --embedder config

# LLM-as-Judge 质量打分
python -m eval_bench.judge_run

# 并发压测（QPS / p95）
python -m eval_bench.loadtest --concurrency 4 --requests 20
```

评测结果默认写入 gitignored 的 `results/`，避免把本地模型输出和个人分析材料发布到仓库。

---

## 致谢

本项目基于 [he-yufeng/CoreCoder](https://github.com/he-yufeng/CoreCoder) 深度二次开发，保留原项目 MIT License 与原作者署名；当前仓库由 [yooumn194](https://github.com/yooumn194) 维护。感谢原作者的开源贡献。

## License

MIT
