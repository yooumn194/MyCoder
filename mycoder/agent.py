"""Core agent loop.

This is the heart of MyCoder.  The pattern is simple:

    user message -> LLM (with tools) -> tool calls? -> execute -> loop
                                      -> text reply? -> return to user

It keeps looping until the LLM responds with plain text (no tool calls),
which means it's done working and ready to report back.
"""

from __future__ import annotations

import concurrent.futures
import inspect
import os
import time
from .llm import LLM
from .memory.integration import MemoryIntegration
from .planner import planning_guard
from .tools import ALL_TOOLS
from .tools.base import Tool
from .tools.agent import AgentTool
from .tools.correction import run_with_correction
from .tools.idempotency import IdempotencyStore
from .tools.subagent_tools import SpawnSubagentTool
from .prompt import system_prompt
from .prompts.reasoning import (
    REASONING_STRATEGIES,
    resolve_strategy,
    resolve_strategy_by_task,
)
from .context import ContextManager
from .tools.security import redact_output


def _injection_guard_enabled() -> bool:
    """True unless the model-layer injection guard is explicitly disabled."""
    return os.getenv("MYCODER_INJECTION_GUARD", "on").strip().lower() != "off"


class Agent:
    def __init__(
        self,
        llm: LLM,
        tools: list[Tool] | None = None,
        max_context_tokens: int = 128_000,
        max_rounds: int = 50,
        memory: MemoryIntegration | None = None,
        tool_selector=None,
        reasoning_strategy: str | None = None,
        injection_detector=None,
        memory_compressor=None,
        experience_store=None,
        budget_guard=None,
    ):
        self.llm = llm
        self.tools = tools if tools is not None else ALL_TOOLS
        self._tool_by_name = {t.name: t for t in self.tools}
        self.messages: list[dict] = []
        self.context = ContextManager(max_tokens=max_context_tokens)
        # P1 memory closure: when a MemoryCompressor is wired, context
        # compression demotes the compressed turns into the long-term memory DB
        # (extract_facts) instead of dropping them.
        if memory_compressor is not None:
            self.context.on_compressed = memory_compressor.on_compressed
        self.max_rounds = max_rounds
        # P1 (prompts/reasoning.py): reasoning strategy — ReAct / Plan-and-
        # Execute / Reflection. A manual override (explicit arg or the
        # PLANNING_STRATEGY env) fixes one strategy for the session; otherwise
        # AUTO mode routes each task to its fitting strategy.
        self._strategy_mode = "auto"
        if reasoning_strategy is not None or os.getenv("PLANNING_STRATEGY"):
            self._strategy_mode = "manual"
        self.reasoning_strategy = resolve_strategy(reasoning_strategy)
        self._system = system_prompt(self.tools, reasoning_strategy=self.reasoning_strategy)
        # Phase 5: optional memory integration (CLI wires one; idempotent).
        self.memory = memory
        if memory is not None:
            memory.install()
        # P1 model-layer injection defense: regex fast-scan on user input AND
        # tool results, plus an LLM classifier on user input (cheapest tier),
        # and <tool_output> role isolation declared in the system prompt.
        # Enabled by default; MYCODER_INJECTION_GUARD=off disables it.
        self._injection = injection_detector
        if self._injection is None and _injection_guard_enabled():
            from .tools.security import InjectionDetector, build_injection_classifier

            self._injection = InjectionDetector(
                classifier=build_injection_classifier(llm)
            )
        # P0 (tools/selector.py): when set, only the tools most relevant to the
        # current user message are injected into the LLM call, cutting token
        # cost and sharpening tool choice. None = inject all (backward compat).
        self.tool_selector = tool_selector
        # P0 (tools/idempotency.py): completed (tool, args) executions, used to
        # serve identical calls from cache instead of re-applying a side effect.
        self._idem = IdempotencyStore()
        # Tool-call metrics: success / failure / retry counters — the material
        # for "成功率 / 工具失败率 / 重试率" answers (see _tool_metrics()).
        self._tool_calls = 0
        self._tool_success = 0
        self._tool_failure = 0
        self._tool_retries = 0
        self._tool_durations: list[float] = []  # ms per real execution

        # wire up sub-agent capability
        for t in self.tools:
            if isinstance(t, AgentTool):
                t._parent_agent = self
            elif isinstance(t, SpawnSubagentTool):
                t._parent_agent = self
                # P1 re-planning experience hook (persists deviation playbooks)
                if experience_store is not None:
                    t._experience_store = experience_store
                # P2 token budget: sub-agents get per-session budget protection
                if budget_guard is not None:
                    t._budget_guard = budget_guard

    def _full_messages(self) -> list[dict]:
        return [{"role": "system", "content": self._system}] + self.messages

    def _select_tools(self, query: str | None = None) -> list[Tool]:
        if self.tool_selector is not None and query and query.strip():
            return self.tool_selector.select(query, self.tools)
        return self.tools

    def _tool_schemas(self, query: str | None = None) -> list[dict]:
        return [t.schema() for t in self._select_tools(query)]

    def _selection_query(self) -> str:
        """Build a tool-selection relevance query from the RECENT conversation,
        so the injected tool set can GROW as the session progresses instead of
        being frozen by the first message for the whole turn (#13)."""
        parts: list[str] = []
        for m in self.messages[-6:]:
            content = m.get("content")
            if isinstance(content, str) and content:
                parts.append(content[:200])
            for tc in m.get("tool_calls") or []:
                if isinstance(tc, dict) and tc.get("name"):
                    parts.append(str(tc["name"]))
        text = "\n".join(parts)[:1500]
        return text or "task"

    def chat(self, user_input: str, on_token=None, on_tool=None) -> str:
        """Process one user message. May involve multiple LLM/tool rounds."""
        # Pre-LLM injection defense: user input is regex-scanned AND LLM-
        # classified (cheapest tier); a hit blocks the message outright.
        if self._injection is not None:
            blocked, reason = self._injection.defend(user_input)
            if blocked:
                return f"⚠ 检测到可能的指令注入，已阻止处理：{reason}"
        # P1 auto mode: route THIS task to its fitting strategy (plan for
        # architecture, reflection for bug-fix, react otherwise) and rebuild the
        # system prompt, so a long session uses the best reasoning mode per task.
        if self._strategy_mode == "auto":
            strategy = resolve_strategy_by_task(user_input)
            if strategy != self.reasoning_strategy:
                self._apply_strategy(strategy)
        self.messages.append({"role": "user", "content": user_input})
        self.context.maybe_compress(self.messages, self.llm)

        for _ in range(self.max_rounds):
            resp = self.llm.chat(
                messages=self._full_messages(),
                tools=self._tool_schemas(self._selection_query()),
                on_token=on_token,
            )

            # no tool calls -> LLM is done, return text
            if not resp.tool_calls:
                self.messages.append(resp.message)
                # Post-LLM: redact secret shapes before the reply reaches the user.
                return redact_output(resp.content)

            # tool calls -> execute (parallel when multiple, like Claude Code's
            # StreamingToolExecutor which runs independent tools concurrently)
            self.messages.append(resp.message)

            try:
                if len(resp.tool_calls) == 1:
                    tc = resp.tool_calls[0]
                    if on_tool:
                        on_tool(tc.name, tc.arguments)
                    result = self._exec_tool(tc)
                    result = self._guard_tool_result(tc.name, result)
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": self._wrap_tool_output(tc.name, result),
                    })
                else:
                    # parallel execution for multiple tool calls
                    results = self._exec_tools_parallel(resp.tool_calls, on_tool)
                    for tc, result in zip(resp.tool_calls, results):
                        result = self._guard_tool_result(tc.name, result)
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": self._wrap_tool_output(tc.name, result),
                        })
            except KeyboardInterrupt:
                # Ctrl+C mid-execution would leave the assistant tool_calls
                # message without replies, poisoning the next request; backfill
                self._answer_pending_tool_calls(resp.tool_calls)
                raise

            # compress if tool outputs are big
            self.context.maybe_compress(self.messages, self.llm)

        return "(reached maximum tool-call rounds)"

    def _guard_tool_result(self, name: str, result: str) -> str:
        """Fast-scan a tool result for injection; a hit replaces it with a
        notice so the malicious text never reaches the model. Only the regex
        scan runs here (tool dumps are large) — the LLM classifier is reserved
        for user input, where it is cheap.
        """
        if self._injection is None:
            return result
        blocked, reason = self._injection.defend(result, use_classifier=False)
        if blocked:
            return f"⚠ 工具 {name} 输出疑似包含指令注入，已隔离：{reason}"
        return result

    @staticmethod
    def _wrap_tool_output(name: str, result: str) -> str:
        """<tool_output> role isolation: mark tool results as untrusted data so
        the system prompt's isolation rule applies to every result."""
        return f'<tool_output tool="{name}">\n{result}\n</tool_output>'

    def _exec_tool(self, tc) -> str:
        """Execute a single tool call, returning the result string."""
        tool = self._tool_by_name.get(tc.name)
        if tool is None:
            return f"Error: unknown tool '{tc.name}'"
        # Phase 3 planning guard: soft/hard interception at the dispatch layer
        # (see mycoder/planner.py). Soft mode preserves open-ended editing;
        # MYCODER_ENFORCE_PLANNING=1 hard-blocks mutation without a plan.
        guard_msg = planning_guard(tc.name, query=tc.arguments.get("task_goal"))
        if guard_msg:
            return guard_msg
        # validate arguments first so a TypeError raised *inside* the tool isn't
        # mislabelled as a bad-arguments error from the caller
        try:
            inspect.signature(tool.execute).bind(**tc.arguments)
        except TypeError as e:
            return f"Error: bad arguments for {tc.name}: {e}"

        # P0 idempotency: an idempotent tool that already completed with the
        # exact same args is served from cache — re-issuing the call must not
        # re-apply a write. Non-idempotent tools are never cached.
        idempotent = bool(getattr(tool, "idempotent", True))
        idem_key = self._idem.key(tool.name, tc.arguments)
        if idempotent:
            cached = self._idem.get(idem_key)
            if cached is not None:
                return cached
        # A real execution (cache miss) — counted for the tool metrics.
        self._tool_calls += 1
        _started = time.monotonic()

        def _count_retry(_strategy):
            self._tool_retries += 1

        try:
            # Phase 3 self-correction: deterministic retry strategies on
            # transient failures (retry_same / retry_modified with timeout
            # extension); everything else surfaces for the agent to reflect.
            # Non-idempotent tools pass retry_safe=False so a side effect that
            # already happened before a failure is never double-applied.
            result = run_with_correction(
                tool.execute,
                retry_safe=idempotent,
                on_retry=_count_retry,
                **tc.arguments,
            )
        except Exception as e:
            self._tool_failure += 1
            self._tool_durations.append((time.monotonic() - _started) * 1000)
            return f"Error executing {tc.name}: {e}"
        if idempotent:
            self._idem.put(idem_key, result)
        if str(result).startswith("Error"):
            self._tool_failure += 1
        else:
            self._tool_success += 1
        self._tool_durations.append((time.monotonic() - _started) * 1000)
        return result

    def _exec_tools_parallel(self, tool_calls, on_tool=None) -> list[str]:
        """Run multiple tool calls concurrently using threads.

        This is inspired by Claude Code's StreamingToolExecutor which starts
        executing tools while the model is still generating.  We simplify to:
        when the model returns N tool calls at once, run them in parallel.

        Duplicate idempotent calls (same tool + args, e.g. a model re-issuing a
        write) are deduped: one thread runs the tool, every duplicate reuses
        its result — otherwise each thread would independently miss the
        idempotency cache and re-apply the side effect (#12).
        """
        from .tools.idempotency import _fingerprint

        for tc in tool_calls:
            if on_tool:
                on_tool(tc.name, tc.arguments)

        # tc -> key. Idempotent duplicates share a key (dedup: one run, all
        # reuse the result); non-idempotent calls get a unique key each.
        tc_key: dict[object, tuple[str, str]] = {}
        order: list[tuple[tuple[str, str], object]] = []
        seen: set[tuple[str, str]] = set()
        for i, tc in enumerate(tool_calls):
            tool = self._tool_by_name.get(tc.name)
            idempotent = bool(getattr(tool, "idempotent", True)) if tool else True
            if idempotent:
                key = (tc.name, _fingerprint(tc.arguments))
                if key not in seen:
                    seen.add(key)
                    order.append((key, tc))
            else:
                key = ("__run__", i)
                order.append((key, tc))
            tc_key[tc] = key

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = {key: pool.submit(self._exec_tool, tc) for key, tc in order}
            results = {key: futures[key].result() for key, _ in order}

        return [results[tc_key[tc]] for tc in tool_calls]

    def _answer_pending_tool_calls(self, tool_calls):
        """Backfill a tool reply for every call that didn't get one.

        OpenAI-compatible APIs reject a request where an assistant message has
        tool_calls without a matching tool reply for each id, so this keeps the
        history valid when execution is interrupted partway through.
        """
        answered = {m.get("tool_call_id") for m in self.messages if m.get("role") == "tool"}
        for tc in tool_calls:
            if tc.id not in answered:
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": "[interrupted]",
                })

    def _apply_strategy(self, strategy: str) -> None:
        """Rebuild the system prompt for a (new) reasoning strategy."""
        self.reasoning_strategy = strategy
        self._system = system_prompt(self.tools, reasoning_strategy=strategy)

    def set_strategy(self, name: str) -> str:
        """Switch the reasoning strategy at runtime.

        'auto'  -> per-task auto routing;
        a strategy name -> fixed manual mode (until 'auto' again).
        Returns a human-readable confirmation message.
        """
        name = (name or "").strip().lower()
        if name == "auto":
            self._strategy_mode = "auto"
            return "推理策略：自动（按任务切换）"
        if name in REASONING_STRATEGIES:
            self._strategy_mode = "manual"
            self._apply_strategy(name)
            return f"推理策略：{name}（手动固定；/strategy auto 切回自动）"
        return f"未知策略 {name!r}；可用: auto|{'|'.join(REASONING_STRATEGIES)}"

    def _tool_metrics(self) -> dict:
        """Tool-call success / failure / retry metrics (面经「成功率、重试率、
        工具失败率」的实测来源). Only real executions count — idempotency-cache
        hits don't re-execute, so they're excluded."""
        total = self._tool_calls
        durations = sorted(self._tool_durations)
        avg_d = sum(durations) / len(durations) if durations else 0.0
        p95_d = (
            durations[min(len(durations) - 1, int(len(durations) * 0.95) - 1)]
            if durations
            else 0.0
        )
        return {
            "calls": total,
            "successes": self._tool_success,
            "failures": self._tool_failure,
            "retries": self._tool_retries,
            "success_rate": round(self._tool_success / total, 4) if total else 0.0,
            "failure_rate": round(self._tool_failure / total, 4) if total else 0.0,
            "retry_rate": round(self._tool_retries / total, 4) if total else 0.0,
            "avg_duration_ms": round(avg_d, 2),
            "p95_duration_ms": round(p95_d, 2),
        }

    def reset(self):
        """Clear conversation history."""
        self.messages.clear()
