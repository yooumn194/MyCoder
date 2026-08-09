<div align="center">

# CoreCoder

**The nanoGPT of coding agents. The main loop is still ~40 lines; the Phase 1–5 production shape around it is ~10,400 lines of Python across 86 files — understand how a coding agent actually works, then fork your own.**

*learn from it · fork it · ship something better*

[中文](README_CN.md) | English | [Source-reading series · 8 bilingual essays](article/00-index_EN.md)

[![PyPI](https://img.shields.io/pypi/v/corecoder)](https://pypi.org/project/corecoder/)
[![Python](https://img.shields.io/badge/python-3.11+-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://github.com/he-yufeng/CoreCoder/actions/workflows/ci.yml/badge.svg)](https://github.com/he-yufeng/CoreCoder/actions)
[![engine](https://img.shields.io/badge/engine-~10.4k_LoC-blue)](article/00-index_EN.md)
[![essays](https://img.shields.io/badge/source--reading-8_bilingual-orange)](article/00-index_EN.md)

</div>

- **Readable end to end.** Every subsystem is plain Python you can follow — sandbox, MCP, multi-agent, memory — with no magic hidden anywhere you can't trace.
- **Hackable.** Set a breakpoint on any line, change it, rerun, all on your own machine. It genuinely works, which makes this a living reference rather than a diagram.
- **The gaps are the point.** It deliberately keeps only the minimal core; what's missing isn't half-finished, it's where you branch off and make it your own.

## How it compares

| | CoreCoder | Claude Code | aider | nanoGPT |
|---|---|---|---|---|
| Lines of code | ~3,600 engine / ~10,400 total | hundreds of thousands (closed) | tens of thousands of Python | ~600 (two files) |
| Time to read it all | one afternoon | can't (closed) | a few days of slogging | one afternoon |
| Breakpoint, change, rerun? | yes, every line | no | yes, but there's a lot | yes |
| What it's for | understand one, then fork your own | production coding assistant | terminal pair-programming | minimal GPT for teaching |

The nanoGPT column is there as a reference point: minimal, readable, but it teaches you to train a GPT. CoreCoder is after the same thing, only the subject is an agent that actually edits code. Sitting it next to Claude Code and aider isn't about competing for their users. CoreCoder is the foundation you stand on while you learn from them and get going; it isn't in the same race.

## What this is

I've always felt coding agents get talked about as if they were arcane. Strip a tool like Claude Code or Cursor all the way down and the core is a `while` loop wrapped around a large model, plus a handful of tools that let it actually do things. The hard part was never the loop; it's everything the loop has to cope with once it meets the real world. CoreCoder is the version that writes that core out honestly, then lets Phases 1–5 grow real-world plumbing around it without hiding any of it.

The core loop (agent, model interface, context, sessions, planning, the tool files) is ~3,600 lines once you drop blank lines and comments — the main loop itself is still about 40. Counting everything Phases 1–5 added — the Docker sandbox, the MCP client, multi-agent orchestration, the result contract, evaluation, and the hybrid-retrieval memory system — the whole package is 86 files: ~10,400 physical lines, 8,502 net, every subsystem still plain Python you can read in a sitting.

And it really runs: reads and writes files, executes shell in a hardened sandbox, speaks MCP, spawns sub-agents, keeps cross-session memory, compacts context in three tiers, and tells you the tokens and dollars a run burned whenever you ask. 447 tests, all green (445 run + 2 container integration tests that skip themselves when Docker is unavailable). But the point of it running isn't to become your daily driver. It runs so the walkthrough can't lie: a reference that shows how an agent works has to actually work.

The code came out of a public teardown: open analyses have already exposed a lot of the load-bearing architecture inside production agents like Claude Code. I took the most essential layer and rewrote it honestly, in as little code as I could. So reading CoreCoder is roughly like reading a runnable, annotated take on how that kind of agent works, except it's only a minimal reimplementation, sitting right there on your machine for you to take apart and change.

<p align="center">
  <img src="https://raw.githubusercontent.com/he-yufeng/CoreCoder/main/assets/demo_en.png" width="760"
       alt="A real CoreCoder run: corecoder -p asks it to fix buggy.py; the agent reads the file, edits the code, runs it to confirm, and reports what it changed.">
</p>

<p align="center"><sub><i>These thousand lines really do run a full loop end to end: ask it to fix buggy.py and it reads the file, edits the code, runs it once to confirm, then reports back on its own. Watch it, then come back and read the code.</i></sub></p>

This README follows the same arc: the first half helps you **read it** (the code map, the main loop, eight essays), the second half helps you **fork it** and points at a few directions worth pushing further.

## Run it once first (five minutes before you read)

Before you read the source, get it running on your machine once to build some intuition. It's a foundation meant for forking, so the recommended path is to clone it and install editable, reading and changing as you go:

```bash
git clone https://github.com/he-yufeng/CoreCoder
cd CoreCoder
pip install -e .
```

If you just want to get it running first, `pip install corecoder` works too.

Give it a model and a key and it goes. It speaks the OpenAI-compatible API by default, and switching providers is usually just two environment variables:

| Provider | Example env vars |
|---|---|
| OpenAI (default `gpt-5.5`) | `OPENAI_API_KEY=sk-...` |
| DeepSeek | `OPENAI_API_KEY=sk-... OPENAI_BASE_URL=https://api.deepseek.com CORECODER_MODEL=deepseek-chat` |
| Local Ollama | `OPENAI_API_KEY=ollama OPENAI_BASE_URL=http://localhost:11434/v1 CORECODER_MODEL=qwen2.5-coder` |

Kimi, Qwen and the like are the same two variables; for providers that don't even offer an OpenAI-compatible endpoint, the optional LiteLLM backend (`pip install "corecoder[litellm]"`) routes to a hundred-plus of them. The third essay goes into this in detail. The key can be `export`ed directly or dropped into a `.env` at the project root, which is loaded on startup. Then:

```bash
corecoder                                             # interactive REPL
corecoder -p "add error handling to parse_config()"   # one-shot mode, exits when done
```

## Code runs in a sandbox

The `bash` tool in the original core gated commands with a regex blacklist — a
handful of known-destructive patterns, trivially bypassed by any command the
list hadn't seen. Phase 1 of the production pass replaces it with
`execute_in_sandbox`: every command runs in a throwaway Docker container that has

- **no network** — there is no channel to exfiltrate to,
- a **read-only root filesystem** and a **non-root `sandbox` user**,
- **no extra privileges** (`no-new-privileges`, every capability dropped),
- **memory / CPU / process caps** (configurable — see the table below) and a
  **hard timeout** — a runaway command kills and re-creates the container,
  keeping the workspace volume intact; if the container keeps OOM-killing,
  execution stops after two retries instead of looping forever,
- the project mounted **read-only at `/src`**, with edits landing in a scratch
  `/workspace` that `get_diff()` surfaces as a unified diff,
- and a **confirmation layer** for risky-but-legal commands (network egress,
  package installs, git rewrites/pushes, recursive deletes) — a lightweight
  mirror of Claude Code's permission system. Approvals are cached per session
  keyed by `(rule, command)`; a confirmation waits at most 60 seconds and
  Ctrl+C counts as a denial; a denied command comes back with a concrete
  alternative and a "don't retry" note. Set `CORECODER_ALLOW_RISKY_COMMANDS=1`
  to auto-approve unattended.

Build the image once:

```bash
docker build -t corecoder-sandbox:3.12 -f sandbox/Dockerfile sandbox/
```

### Resource limits

| Env var | Default | Meaning |
|---|---|---|
| `CORECODER_SANDBOX_MEM` | `512m` | container memory cap |
| `CORECODER_SANDBOX_CPU` | `0.5` | CPU cores (may be fractional) |
| `CORECODER_SANDBOX_PIDS` | `128` | max processes in the container |

Large front-end builds: `CORECODER_SANDBOX_MEM=2g`. Java/Maven compiles:
`CORECODER_SANDBOX_MEM=4g CORECODER_SANDBOX_CPU=2`.

### Sandbox → host file sync

The sandbox writes to `/workspace`, which the host file tools can't see.
`execute_in_sandbox` therefore only *reports* which files changed (up to 50,
via `git status` in the workspace), and `sync_workspace()` is the explicit
pull-back step:

- `sync_workspace()` copies changed files to the host project directory.
- `sync_workspace(clean=True)` additionally deletes host files that were
  deleted inside the sandbox.
- `read_file` / `write_file` accept `/workspace/...` paths directly: when the
  sandbox volume exists they are mapped onto the host project directory. The
  mapping is keyed on the **volume**, not the container, so it still works
  after the sandbox is stopped.

If Docker isn't reachable the agent degrades gracefully instead of dying: it
**fails closed** by default, and only runs commands on the host with an
allowlist, a timeout, and a WARNING audit log after you confirm — set
`CORECODER_ALLOW_LOCAL_EXEC=1` for unattended opt-in.

### Phase 2: pure-tool search

The agent finds code by *searching*, not by a hand-built index. Phase 2 added
the agentic-search pair plus the gate both share:

- **`grep_search`** runs ripgrep when it's on `PATH` and falls back to a
  pure-Python scan otherwise (10–50× slower on huge trees, identical output).
  **`list_files`** finds files by glob. Both are path-guarded and symlink-safe.
- **`path_guard`** is the single gate in front of them: no traversal out of the
  workspace, no symlink escapes (a macOS `/var` symlinked-root gotcha is
  handled), before any tool touches the filesystem.
- **`prompts/search_strategy.py`** teaches the agent *how* to search — narrow
  with targeted greps instead of reading whole files — so exploration stays
  cheap even on large repos. Zero index, zero vectors: the whole strategy is
  pure tools.

### Phase 3: plan, correct, route

The Phase 3 pass turns the agent from an excellent executor into a deliberate
decision-maker:

- **`todo_write` / `todo_update`** create a structured plan and mark steps
  `in_progress` / `done`. Invalid plans (cycles, dangling dependencies) are
  rejected before they pollute context.
- **A planning guard** in the dispatch layer blocks a mutation tool when the
  current step isn't `in_progress` (once a plan exists); set
  `CORECODER_ENFORCE_PLANNING=1` to also block mutations when no plan exists.
- **Self-correction** classifies tool failures deterministically (transient →
  retry with a longer timeout, OOM → fail fast, permission → escalate to the
  user) instead of blindly retrying.
- **`ModelRouter`** maps a task to a model tier via `config/model_routing.yaml`
  (hot-reloaded; override the tier with `CORECODER_MODEL_TIER`).
- **MCP Lite** is a pre-wired stdio MCP client whose timeouts raise a
  structured error the correction loop understands (Phase 3.5 wires real
  servers).

Other tunables: `CORECODER_CONFIRM_TIMEOUT` (default `60`, the confirmation
prompt deadline) and, for `grep_search`, note that without `rg` the pure-Python
fallback is used — roughly 10–50× slower on large trees, identical output.

### Phase 3.5: MCP

`corecoder/mcp/` exposes external MCP servers as ordinary tools
(`mcp_<server>_<tool>`), protocol-isolated: JSON-RPC framing, SSE reconnect,
capability negotiation and error-code mapping are all digested behind
`Tool.execute(**kwargs) -> str`, so Planning and Self-Correction need no
changes. Two transports — stdio (Content-Length framing, crash restart, stderr
correlated to the active request) and SSE (dual-endpoint discovery,
Last-Event-ID replay, exponential-backoff reconnect) — plus a security policy
(per-server tool whitelist, parameter regexes like `^/workspace/.*`), secrets
via `token_env`, discovery timeouts (`skip`/`partial`/`block`), and a
per-call structured trace. Configure servers in `config/mcp_servers.yaml`
(all opt-in; the CLI picks them up automatically). Migration: the Phase 3
`tools/mcp_lite.py` prototype is superseded by `corecoder.mcp`; its
`MCPToolError` is now the unified type.

**Dependency note.** The MCP layer adds zero mandatory dependencies: the stdio
transport is pure standard library. SSE needs `aiohttp` — install it with
`pip install corecoder[mcp]` (imported lazily, so you only pay for it if you
use SSE). aiohttp over httpx: SSE streaming support is more mature and
connection reuse is easier to control; a future switch to httpx + anyio is a
low-risk, low-priority option.

**Migrate from SSE to Streamable HTTP** (MCP 2025-03-26 supersedes SSE).

| Old (SSE) | New (Streamable HTTP) |
|---|---|
| `transport: sse` + `sse_endpoint` / `post_endpoint` | `transport: streamable_http` + a single `endpoint` |

```yaml
# old
servers:
  github:
    transport: sse
    sse_endpoint: https://mcp.github.com/sse
    post_endpoint: https://mcp.github.com/messages
# new
servers:
  github:
    transport: streamable_http
    endpoint: https://mcp.github.com/api/v1
```

Behavioral differences: SSE uses a separate GET event stream + POST requests;
Streamable HTTP uses one POST endpoint whose response body IS the SSE stream
(and supports the 202 Accepted semantic). The SSE transport still works but
logs a deprecation warning. Timeline: deprecated in v0.4.x, disabled by
default in v0.5.0, removed in v1.0.0.

### Phase 4: system intelligence

CoreCoder grows from a single smart agent into an orchestratable agent system:

- **Multi-agent orchestration** (`corecoder/agents/`): `SubagentDefinition`
  (explorer / planner / implementer / reviewer), `SubagentRunner` (isolated
  context, timeout, token budget, returns a validated RFC v1.0.1 envelope),
  `Blackboard` (shared KV store with TTL and `asyncio.Lock`), and an
  `Orchestrator` (sequential / parallel / conditional strategies, circuit
  breaker after 3 consecutive failures). The `spawn_subagent` tool exposes it
  to the main agent.
- **Subagent Result Contract** (`corecoder/contracts/`): the frozen RFC v1.0.1
  envelope enforced by Pydantic — state-combination matrix, strict
  `completeness_ratio` bounds, per-instance idempotency UUIDs, artifact caps.
- **LSP symbol intelligence**: LSP servers integrate via MCP (`mcp-server-lsp`,
  opt-in in `config/mcp_servers.yaml`), with intent-aware tool descriptions
  (✅/❌ scenarios) and an `LSPResultCompressor` (dedup → rank → truncate).
- **Streamable HTTP transport** (MCP 2025-03-26): POST response body IS the
  SSE stream, coexists with the existing SSE transport.
- **Evaluation** (`corecoder/eval/`): orchestration metrics (delegation
  accuracy, speedup, context inflation, LSP adoption), a failure-pattern
  knowledge base, and an incremental verification dashboard.

### Phase 5: hybrid-retrieval memory

Cross-session memory that survives restarts, with **zero infrastructure** and
**graceful degradation** (every optional backend can be missing):

- **Storage** (`corecoder/memory/store.py`): two SQLite databases with the same
  schema — project (`<repo>/.corecoder/memory.db`) and global
  (`~/.corecoder/memory.db`). Each holds a `memories` table, a manually-managed
  FTS5 index (`tokenize='ascii'`, content is jieba/bigram-tokenized before
  insert), and an `embeddings` vector table.
- **Tokenization** (`corecoder/memory/tokenizer.py`): jieba word segmentation
  when installed, otherwise a zero-dependency CJK bigram tokenizer — both give
  *word-level* Chinese matching (searching `认证模块` finds a memory containing
  `认证模块使用JWT…`), never single-character.
- **Embeddings** (`corecoder/memory/embedder.py`): `fastembed` →
  `sentence-transformers` → built-in deterministic hashing backend (numpy), or
  `none`. Heavy models load lazily; every backend shares a bounded hand-rolled
  LRU cache.
- **Vectors** (`store._NumpyVectorBackend` / `_Vec0VectorBackend`): `sqlite-vec`
  when the extension loads, otherwise brute-force cosine over a BLOB column —
  so the hybrid path works with zero extra dependencies.
- **Hybrid retrieval** (`corecoder/memory/retriever.py`): BM25 (FTS5, both
  dbs) + vector cosine fused with Reciprocal Rank Fusion (`k=60`); filters
  (scope / type / min confidence / deprecated) are applied with a batched
  `IN` query.
- **Confidence decay** (`corecoder/memory/maintenance.py`): `auto` memories
  untouched for 30 days (≥3 accesses) lose confidence — project ×0.8, global
  ×0.95 — and drop below a threshold become `deprecated_by='decayed'`, pruned
  by `compact()`. `user` / `confirmed` memories never decay.
- **Six tools**: `memory_save` / `memory_search` / `memory_list` /
  `memory_forget` / `memory_confirm` / `memory_stats`. Saves are deduplicated
  (cosine > 0.85 → update, not duplicate) and redacted for secrets.
- **Integration**: before `todo_write`, relevant memories are injected into the
  plan prompt (`planning_guard`); a recovered Self-Correction failure settles a
  `pattern` memory; a finished plan is distilled into a `decision` memory.

Configuration lives in `config/memory.yaml` (`embedder.backend`, `rrf_k`,
`max_tokens`, `decay_days`, …). All of it is optional and lazily loaded.

## Read it: the code map

Laid out flat, the whole project is this big. Skim it before you clone and you'll know where everything is. This is the most concrete difference from Claude Code's hundreds of thousands of lines: you can read it like the table of contents of a book. Start from the main loop in `agent.py`; that's the heart of the whole agent.

```
corecoder/
├── agent.py          agent loop + parallel tool exec       172 lines   ← start here
├── llm.py            streaming client + retry + cost        344 lines
├── context.py        three-tier context compaction          210 lines
├── session.py        save / resume + path-traversal guard    97 lines
├── prompt.py         system prompt + search strategy         39 lines
├── prompts/          reusable prompt segments                36 lines   ← Phase 2
├── cli.py            REPL + slash commands + one-shot       306 lines
├── config.py         env-var config                          57 lines
├── planner.py        planning engine (Todo/Plan/Guard)      298 lines   ← Phase 3
├── model_router.py   task→model-tier routing (YAML)         104 lines   ← Phase 3
├── sandbox/          Docker-backed command isolation        1568 lines  ← Phase 1
│   ├── docker_executor.py  hardened container lifecycle     495 lines
│   ├── executor.py         backend selection + fallback     229 lines
│   ├── sync.py             /workspace <-> host sync         255 lines
│   ├── policy.py           permission confirm + cache       284 lines
│   ├── local_executor.py   degraded host allowlist          160 lines
│   └── models·logger·locking·__init__                       145 lines
├── mcp/              MCP client (stdio/SSE/Streamable)      1623 lines  ← Phase 3.5/4
│   ├── client.py     transports + retry                      92 lines
│   ├── registry.py   server loading + tool registration      166 lines
│   ├── adapter.py    MCP tool adapters                       131 lines
│   ├── security.py   per-server allowlist + param regexes     44 lines
│   ├── lsp_metadata.py · lsp_compressor.py  LSP intelligence  225 lines  ← Phase 4
│   └── config·errors·observability·runtime·__init__          203 lines
├── agents/           multi-agent orchestration               819 lines  ← Phase 4
│   ├── orchestrator.py  seq/parallel/conditional + breaker   306 lines
│   ├── runner.py        sub-agent executor                   272 lines
│   ├── definition.py    SubagentDefinition                    78 lines
│   ├── blackboard.py    shared KV + TTL                       60 lines
│   └── tool_validator.py sub-agent tool validation            87 lines
├── contracts/        RFC envelope + Pydantic validation      609 lines  ← Phase 4
│   ├── subagent_result.py  envelope + state-combination matrix  321 lines
│   └── envelope.py · prompts.py                              288 lines
├── eval/             evaluation harness (metrics·runner·kb·dashboard)  226 lines  ← Phase 4
├── memory/           hybrid-retrieval memory system          1588 lines  ← Phase 5
│   ├── store.py      dual-db SQLite + FTS5 + vector backends  619 lines
│   ├── embedder.py   multi-backend embeddings + LRU           198 lines
│   ├── retriever.py  BM25 + vector, RRF fusion                103 lines
│   ├── maintenance.py confidence decay + compact + stats      113 lines
│   ├── integration.py planning_guard / Self-Correction wiring 130 lines
│   ├── types.py      data models                              129 lines
│   ├── tokenizer.py  jieba→bigram degraded tokenization        75 lines
│   ├── prompt.py     memory-section injection + token budget   74 lines
│   └── config·security·__init__                               147 lines
└── tools/            twenty tools                             2299 lines
    ├── sandbox_tool.py  execute_in_sandbox (replaces bash)    184 lines
    ├── sync_tool.py     sync_workspace (pull changes back)     76 lines
    ├── grep_search.py   rg-first regex search + fallback       232 lines  ← Phase 2
    ├── list_files.py    glob file lookup (symlink-safe)         96 lines  ← Phase 2
    ├── path_guard.py    shared path-traversal/symlink gate      78 lines  ← Phase 2
    ├── read_file.py     read + ranges + 300-line cap           127 lines
    ├── todo_tools.py    todo_write / todo_update (planning)    144 lines  ← Phase 3
    ├── correction.py    error→strategy self-correction         169 lines  ← Phase 3
    ├── mcp_lite.py      MCP prototype client                   121 lines  ← Phase 3
    ├── subagent_tools.py spawn_subagent                         83 lines  ← Phase 4
    ├── memory_tools.py  six memory tools                       199 lines  ← Phase 5
    ├── fetch.py         fetch_url                               40 lines
    ├── workspace_path.py  /workspace path mapping               57 lines
    ├── bash.py          pre-check regex gate (kept as helper)  127 lines
    ├── edit.py          unique-match search/replace + diff      92 lines
    ├── grep.py          content search (legacy)                 84 lines
    ├── glob_tool.py     filename matching (legacy)              52 lines
    ├── batch_diagnostics.py  sandbox diagnostics helper          41 lines
    ├── write.py         file write                              45 lines
    ├── agent.py         sub-agent spawning                     162 lines
    └── base.py          tool base class                         27 lines
```

(The container image itself is built from `sandbox/Dockerfile` at the repo root.) Twenty tools: `execute_in_sandbox` (the sandboxed successor to `bash`), `sync_workspace`, `grep_search` and `list_files` (the Phase 2 agentic-search pair: zero index, zero embedding, path-guarded, ripgrep-first with a pure-Python fallback), `todo_write` and `todo_update` (the Phase 3 planning pair), `spawn_subagent` (Phase 4), the Phase 5 memory tools `memory_save` / `memory_search` / `memory_list` / `memory_forget` / `memory_confirm` / `memory_stats`, and the Phase 1/2 file tools `read_file`, `write_file`, `edit_file`, `glob`, `grep`, `fetch_url`, `agent` (which spawns a sub-agent). The search strategy lives in `prompts/search_strategy.py`; the model-routing rules in `config/model_routing.yaml`; MCP servers in `config/mcp_servers.yaml` (all opt-in); the memory config in `config/memory.yaml`. Everything else is the CLI shell, config, and packaging wrapped around that engine core.

## A `while` loop is the whole agent

The whole of an agent fits in one sentence: hand the user's words to the model, run whatever tools it asks for, stuff the results back into the context, ask again, and keep going until it stops asking for tools and gives an answer. In code, that's about a dozen lines:

```python
# corecoder/agent.py · the main loop (trimmed skeleton)
def chat(self, user_input):
    self.messages.append(user_input)

    for _ in range(self.max_rounds):                   # bounded, so it can't run away
        reply = self.llm.chat(self.messages, self.tools)   # ask the model what to do next
        if not reply.tool_calls:                       # model wants no more tools
            return reply.text                          #   -> done, hand the answer back
        results = run_parallel(reply.tool_calls)       # tools requested -> run in parallel
        self.messages += results                       # feed results back, loop again

    return "(hit the round limit)"
```

That's the whole thing. The core skeleton is about twenty lines; counting parallel execution and the bookkeeping after a Ctrl+C interrupt, maybe forty. Almost everything else in CoreCoder's ~10,000 lines is there to clean up the mess the loop runs into once it meets the real world. `llm.py` — now 344 lines, among the biggest engine files — got that way not because calling a model is hard, but because a streamed response splinters each tool call's arguments into fragments you have to restitch in order, a provider will hand you half a JSON object or a null `usage` field, and 429s, timeouts, dropped connections and 5xx all need backoff-and-retry while the other 4xx should just raise. That unglamorous grunt work, not the loop, is where the real engineering of taking an agent from demo to delivery actually lives; the third essay follows it down to the line.

Three decisions are worth a closer look, because they're the kind of call you can only make after you've understood how others did it, and they're judgments you can lift straight into your own fork.

**`edit_file` does search-and-replace on a unique match, not line numbers.** Line numbers are a trap: the model only has to miscount by one and it quietly edits the wrong place. Anchor on a unique snippet of the original instead. If there's no match, it hands the start of the file back so the model can re-anchor; if there are several matches, it makes the model bring more surrounding context rather than gamble on one. On a successful edit it returns a diff. Recoverable on failure, verifiable on success: the whole loop stays inside the tool.

**Context isn't cut all at once when it's full; it gives ground in three tiers, cheapest first.** At half full (50%) it trims over-long tool outputs in place, a tier that's purely mechanical and costs no model call. If 70% still isn't enough, it has the model summarize the older turns into a single paragraph while keeping the most recent ones verbatim. Only at 90% does it hit the emergency tier and pull everything, summary and recent turns alike, down to its tightest form. Blunt truncation tends to throw away exactly the early decision a long task leans on most; tiering lets it surrender the least important things first instead of lopping off the oldest decisions wholesale from the start.

**You constrain a sub-agent by withholding the tool, not by writing rules and hoping it obeys.** A spawned sub-agent gets an isolated context and its own separate history, with a toolset exactly one item shorter than the parent's: the `agent` tool itself, so it can't recursively spawn more sub-agents. Handing it one fewer tool is cleaner than legislating a rule after the fact. It also reuses the parent's model connection (its spend folded into the same running total), truncates its output once it runs past 5,000 characters down to just the opening, and runs on a shorter round limit than the parent. The same restraint, end to end.

Every one of these *whys* is traced down to the actual lines of code in the series below.

## The source-reading series · 8 bilingual essays

I also wrote a bilingual source-reading series, one intro plus seven parts, each in Chinese with an English mirror. Against CoreCoder's actual code, it walks through how agents like Claude Code work under the hood. One hard rule I set myself: every line count and every snippet is re-read and re-checked from the repo, never written from memory. The first six get you reading, the seventh gets you forking; read them in any order.

- **[Intro · Read Claude Code through CoreCoder, then build your own](article/00-index_EN.md)**
- **[01 · An agent, at its core, is a `while` loop](article/01-the-loop_EN.md)** — the main loop in `agent.py`, interrupts, and the round limit
- **[02 · The tool system: letting the model act, safely](article/02-tools_EN.md)** — the seven tools in `tools/` and the bash safety gate
- **[03 · Plug in any LLM, and keep the bill honest](article/03-llm-and-cost_EN.md)** — `llm.py`'s provider wrapper, retries, and cost accounting
- **[04 · Surviving a long task on a finite window](article/04-context_EN.md)** — `context.py`'s three-tier compaction and orphaned tool messages
- **[05 · Parallel execution and sub-agents](article/05-parallel-and-subagents_EN.md)** — thread-pool concurrency and sub-agent isolation
- **[06 · Turning it into a real command-line tool](article/06-session-and-cli_EN.md)** — `session.py` and path-traversal defense
- **[07 · Fork CoreCoder into your own coding agent](article/07-build-your-own_EN.md)** — from fork to custom tools to swapping models

## Fork it, build something better

Once you understand it, the natural next step is to fork. Getting started doesn't take much:

- **Swap in a model you actually use.** It's the two env vars from above; `llm.py` (344 lines) is the entry point for all provider adaptation.
- **Add a tool of your own.** Write a new file against the tool base class in `tools/base.py` (27 lines): run tests, fetch a page, call an LSP, whatever. The end of the second essay walks you through your first one by hand.
- **Rewrite the system prompt.** `prompt.py` is all of 39 lines; change one line and you'll watch the agent's temperament shift. It's the cheapest "change one thing, see a result" in the whole project.
- **Import it as a library.** The top level exports `Agent`, `LLM`, and `Config`, ready to embed in your own program:

```python
from corecoder import Agent, LLM

llm = LLM(model="deepseek-chat", api_key="sk-...", base_url="https://api.deepseek.com")
print(Agent(llm=llm).chat("find every TODO comment in this project and list them"))
```

Going deeper, the directions are out in the open too. None of the following is in CoreCoder, by design, not because it's unfinished. Flip it around and each one is an entry point you can carry into a real tool of your own:

- **The sandbox isolates the shell, not the whole agent.** `execute_in_sandbox` runs commands in a hardened container (no network, read-only rootfs, caps dropped, limits enforced), but file edits still land on your host through `edit_file`/`write_file`, and the sandbox workspace is surfaced as a diff rather than written back. Making file tools sandboxed too — or syncing the workspace out on exit — is the natural next step.
- **Retry is only exponential backoff.** No fallback model, no hard dollar budget. Follow `llm.py` down and add a fallback model chain plus a stop-on-over-budget gate; the change stays mostly inside that one file.
- **Sub-agent execution is orchestrated but not streamed.** Phase 4 added an async `Orchestrator` and validated envelopes, but the main agent's `spawn_subagent` still runs synchronously with truncated output. A streaming/async executor — the main agent keeps working while a sub-agent streams — closes the exact gap the fifth essay identifies between this and how production agents stream execution.
- **Memory is a lean local retriever, not a full RAG index.** Phase 5's hybrid search is SQLite FTS5 + numpy vectors fused by RRF — great for cross-session notes, not a chunked-and-embedded index over a big repo. Wiring an external vector DB or code chunking over a large codebase is the natural next step.
- **MCP is a client, not a marketplace.** Phase 3.5/4 give it a real MCP client (stdio/SSE/Streamable HTTP) and LSP intelligence, but servers are still configured by hand. Auto-discovering servers, or packaging the agent's own tools as an MCP server for other agents, are both open.

The README only points; the seventh essay picks up the code details for each. Pick one and start; that's the whole reason the core is kept this small.

## Commands

Inside the REPL, `/help` lists everything; these are the ones you'll reach for:

```
/model <name>    switch model
/compact         compact the context by hand
/tokens          token usage and cost estimate
/diff            files changed this session
/save  /sessions save / list sessions
quit / exit      exit (Ctrl+C cancels the current round)
```

Session IDs are sanitized to safe characters before they become filenames, every archive lands under `~/.corecoder/sessions`, and a malicious session name can't traverse out.

## Related Projects

If working through CoreCoder was useful, here are a few other tools I've built around agents and LLM systems:

- **[RepoWiki](https://github.com/he-yufeng/RepoWiki)** — dropped into an unfamiliar codebase? It gives you a guided wiki and a where-to-start reading path, a self-hostable DeepWiki alternative.
- **[FindJobs-Agent](https://github.com/he-yufeng/FindJobs-Agent)** — stop sifting job boards by hand: it ranks postings against your resume and runs mock interviews.
- **[ContractGuard](https://github.com/he-yufeng/ContractGuard)** — catch the risky clauses before you sign: it reads contracts and flags the dangerous bits.
- **[GitSense](https://github.com/he-yufeng/GitSense)** — want to contribute to open source? It finds issues worth your time and gauges whether your PR will get merged.
- **[CodeABC](https://github.com/he-yufeng/CodeABC)** — understand any codebase even if you don't code, built for non-programmers.

## Contributing / License

Before you send anything, run `pytest tests/ -q` (447 tests: 445 run + 2 Docker-gated skips), `ruff check`, and `compileall`, and make sure they're green. The Docker-backed sandbox tests need the image built once: `docker build -t corecoder-sandbox:3.12 -f sandbox/Dockerfile sandbox/`. MIT licensed: fork it, learn from it, ship something better. A mention of this project is appreciated.

---

By [Yufeng He](https://github.com/he-yufeng), formerly at Moonshot AI (Kimi). I earlier wrote a fairly complete [Claude Code source analysis](https://zhuanlan.zhihu.com/p/1898797658343862272) on Zhihu; this project is its hands-on counterpart: that one walks you through reading it, this one through rebuilding it.

> CoreCoder was formerly named NanoCoder; it was renamed to avoid confusion with [Nano-Collective/nanocoder](https://github.com/Nano-Collective/nanocoder), and old links redirect here automatically.
