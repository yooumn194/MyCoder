<div align="center">

# CoreCoder

**The nanoGPT of coding agents. 1,081 lines of pure Python — understand how a coding agent actually works, then fork your own.**

*learn from it · fork it · ship something better*

[中文](README_CN.md) | English | [Source-reading series · 8 bilingual essays](article/00-index_EN.md)

[![PyPI](https://img.shields.io/pypi/v/corecoder)](https://pypi.org/project/corecoder/)
[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://github.com/he-yufeng/CoreCoder/actions/workflows/ci.yml/badge.svg)](https://github.com/he-yufeng/CoreCoder/actions)
[![engine](https://img.shields.io/badge/engine-1081_LoC-blue)](article/00-index_EN.md)
[![essays](https://img.shields.io/badge/source--reading-8_bilingual-orange)](article/00-index_EN.md)

</div>

- **Readable end to end.** Read the whole engine in an afternoon, with no magic hidden anywhere you can't follow it.
- **Hackable.** Set a breakpoint on any line, change it, rerun, all on your own machine. It genuinely works, which makes this a living reference rather than a diagram.
- **The gaps are the point.** It deliberately keeps only the minimal core; what's missing isn't half-finished, it's where you branch off and make it your own.

## How it compares

| | CoreCoder | Claude Code | aider | nanoGPT |
|---|---|---|---|---|
| Lines of code | ~1,081 engine / 1,714 total | hundreds of thousands (closed) | tens of thousands of Python | ~600 (two files) |
| Time to read it all | one afternoon | can't (closed) | a few days of slogging | one afternoon |
| Breakpoint, change, rerun? | yes, every line | no | yes, but there's a lot | yes |
| What it's for | understand one, then fork your own | production coding assistant | terminal pair-programming | minimal GPT for teaching |

The nanoGPT column is there as a reference point: minimal, readable, but it teaches you to train a GPT. CoreCoder is after the same thing, only the subject is an agent that actually edits code. Sitting it next to Claude Code and aider isn't about competing for their users. CoreCoder is the foundation you stand on while you learn from them and get going; it isn't in the same race.

## What this is

I've always felt coding agents get talked about as if they were arcane. Strip a tool like Claude Code or Cursor all the way down and the core is a `while` loop wrapped around a large model, plus seven or eight tools that let it actually do things. The hard part was never the loop; it's everything the loop has to cope with once it meets the real world. CoreCoder is the minimal version that writes that core out honestly.

The engine (loop, model interface, context, tools, sessions) is 1,081 lines once you drop blank lines and comments. Counting the outer CLI, config and packaging too, the whole package is 18 files: 1,714 physical lines, 1,385 net, every one short enough to read in a single sitting.

And it really runs: reads and writes files, executes shell in a sandbox, spawns sub-agents, compacts context in three tiers, and tells you the tokens and dollars a run burned whenever you ask. 398 tests, all green (the container integration tests skip themselves when Docker is unavailable). But the point of it running isn't to become your daily driver. It runs so the walkthrough can't lie: a reference that shows how an agent works has to actually work.

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

## Read it: the code map

Laid out flat, the whole project is this big. Skim it before you clone and you'll know where everything is. This is the most concrete difference from Claude Code's hundreds of thousands of lines: you can read it like the table of contents of a book. Start from the main loop in `agent.py`; that's the heart of the whole agent.

```
corecoder/
├── agent.py        agent loop + parallel tool exec       155 lines   ← start here
├── llm.py          streaming client + retry + cost        336 lines
├── context.py      three-tier context compaction          210 lines
├── session.py      save / resume + path-traversal guard    97 lines
├── prompt.py       system prompt + search strategy         40 lines
├── prompts/        reusable prompt segments                35 lines   ← Phase 2
├── cli.py          REPL + slash commands + one-shot       270 lines
├── config.py       env-var config                          57 lines
├── planner.py      planning engine (Todo/Plan/Guard)      243 lines   ← Phase 3
├── model_router.py task→model-tier routing (YAML)         104 lines   ← Phase 3
├── sandbox/        Docker-backed command isolation        ~1200 lines   ← Phase 1
│   ├── docker_executor.py  hardened container lifecycle   440 lines
│   ├── executor.py         backend selection + fallback   230 lines
│   ├── sync.py             /workspace <-> host sync       160 lines
│   ├── policy.py           permission confirm + cache    230 lines
│   ├── local_executor.py   degraded host allowlist        160 lines
│   └── logger.py·models.py·locking.py                      90 lines
└── tools/
    ├── sandbox_tool.py  execute_in_sandbox (replaces bash) 165 lines
    ├── sync_tool.py     sync_workspace (pull changes back)  90 lines
    ├── grep_search.py   rg-first regex search + fallback    230 lines   ← Phase 2
    ├── list_files.py    glob file lookup (symlink-safe)      90 lines    ← Phase 2
    ├── read_file.py     read + ranges + 300-line cap         140 lines
    ├── path_guard.py    shared path-traversal/symlink gate    70 lines   ← Phase 2
    ├── todo_tools.py    todo_write / todo_update (planning)  128 lines   ← Phase 3
    ├── correction.py    error→strategy self-correction       125 lines   ← Phase 3
    ├── mcp_lite.py      MCP client + structured timeouts     128 lines   ← Phase 3
    ├── workspace_path.py  /workspace path mapping          55 lines
    ├── bash.py       pre-check regex gate (kept as helper) 127 lines
    ├── edit.py       unique-match search/replace + diff    92 lines
    ├── grep.py       content search (legacy)               79 lines
    ├── glob_tool.py  filename matching (legacy)            47 lines
    ├── write.py      file write                            38 lines
    ├── agent.py      sub-agent spawning                    58 lines
    └── base.py       tool base class                       27 lines
```

(The container image itself is built from `sandbox/Dockerfile` at the repo root.) Thirteen tools: `execute_in_sandbox` (the sandboxed successor to `bash`), `sync_workspace`, `grep_search` and `list_files` (the Phase 2 agentic-search pair: zero index, zero embedding, path-guarded, ripgrep-first with a pure-Python fallback), `todo_write` and `todo_update` (the Phase 3 planning pair), `read_file`, `write_file`, `edit_file`, `glob`, `grep`, `agent` (which spawns a sub-agent), and `fetch_url`. The search strategy lives in `prompts/search_strategy.py`; the model-routing rules in `config/model_routing.yaml`. Everything else is the CLI shell, config, and packaging wrapped around that engine core.

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

That's the whole thing. The core skeleton is about twenty lines; counting parallel execution and the bookkeeping after a Ctrl+C interrupt, maybe forty. Almost everything else in CoreCoder's thousand-odd lines is there to clean up the mess the loop runs into once it meets the real world. `llm.py` ends up the biggest file in the project, not because calling a model is hard, but because a streamed response splinters each tool call's arguments into fragments you have to restitch in order, a provider will hand you half a JSON object or a null `usage` field, and 429s, timeouts, dropped connections and 5xx all need backoff-and-retry while the other 4xx should just raise. That unglamorous grunt work, not the loop, is where the real engineering of taking an agent from demo to delivery actually lives; the third essay follows it down to the line.

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

- **Swap in a model you actually use.** It's the two env vars from above; `llm.py` (336 lines) is the entry point for all provider adaptation.
- **Add a tool of your own.** Write a new file against the tool base class in `tools/base.py` (27 lines): run tests, fetch a page, call an LSP, whatever. The end of the second essay walks you through your first one by hand.
- **Rewrite the system prompt.** `prompt.py` is all of 33 lines; change one line and you'll watch the agent's temperament shift. It's the cheapest "change one thing, see a result" in the whole project.
- **Import it as a library.** The top level exports `Agent`, `LLM`, and `Config`, ready to embed in your own program:

```python
from corecoder import Agent, LLM

llm = LLM(model="deepseek-chat", api_key="sk-...", base_url="https://api.deepseek.com")
print(Agent(llm=llm).chat("find every TODO comment in this project and list them"))
```

Going deeper, the directions are out in the open too. None of the following is in CoreCoder, by design, not because it's unfinished. Flip it around and each one is an entry point you can carry into a real tool of your own:

- **The sandbox isolates the shell, not the whole agent.** `execute_in_sandbox` runs commands in a hardened container (no network, read-only rootfs, caps dropped, limits enforced), but file edits still land on your host through `edit_file`/`write_file`, and the sandbox workspace is surfaced as a diff rather than written back. Making file tools sandboxed too — or syncing the workspace out on exit — is the natural next step.
- **Retry is only exponential backoff.** No fallback model, no hard dollar budget. Follow `llm.py` down and add a fallback model chain plus a stop-on-over-budget gate; the change stays mostly inside that one file.
- **Sub-agents only run the plainest synchronous execution.** Make it async or a streaming executor and you close the exact gap the fifth essay identifies between this and how production agents stream execution.
- **No MCP, no RAG.** Wire up MCP to give it the external tool ecosystem, or add retrieval-based code location for big repos. Both are real ways to grow from a minimal core into your own stronger agent.

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

Before you send anything, run `pytest tests/ -q` (398 tests), `ruff check`, and `compileall`, and make sure they're green. The Docker-backed sandbox tests need the image built once: `docker build -t corecoder-sandbox:3.12 -f sandbox/Dockerfile sandbox/`. MIT licensed: fork it, learn from it, ship something better. A mention of this project is appreciated.

---

By [Yufeng He](https://github.com/he-yufeng), formerly at Moonshot AI (Kimi). I earlier wrote a fairly complete [Claude Code source analysis](https://zhuanlan.zhihu.com/p/1898797658343862272) on Zhihu; this project is its hands-on counterpart: that one walks you through reading it, this one through rebuilding it.

> CoreCoder was formerly named NanoCoder; it was renamed to avoid confusion with [Nano-Collective/nanocoder](https://github.com/Nano-Collective/nanocoder), and old links redirect here automatically.
