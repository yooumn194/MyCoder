"""Core agent loop.

This is the heart of CoreCoder.  The pattern is simple:

    user message -> LLM (with tools) -> tool calls? -> execute -> loop
                                      -> text reply? -> return to user

It keeps looping until the LLM responds with plain text (no tool calls),
which means it's done working and ready to report back.
"""

from __future__ import annotations

import concurrent.futures
import inspect
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
from .prompts.reasoning import resolve_strategy
from .context import ContextManager


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
    ):
        self.llm = llm
        self.tools = tools if tools is not None else ALL_TOOLS
        self._tool_by_name = {t.name: t for t in self.tools}
        self.messages: list[dict] = []
        self.context = ContextManager(max_tokens=max_context_tokens)
        self.max_rounds = max_rounds
        # P1 (prompts/reasoning.py): explicit reasoning strategy — ReAct /
        # Plan-and-Execute / Reflection, switchable via PLANNING_STRATEGY.
        self.reasoning_strategy = resolve_strategy(reasoning_strategy)
        self._system = system_prompt(self.tools, reasoning_strategy=self.reasoning_strategy)
        # Phase 5: optional memory integration (CLI wires one; idempotent).
        self.memory = memory
        if memory is not None:
            memory.install()
        # P0 (tools/selector.py): when set, only the tools most relevant to the
        # current user message are injected into the LLM call, cutting token
        # cost and sharpening tool choice. None = inject all (backward compat).
        self.tool_selector = tool_selector
        # P0 (tools/idempotency.py): completed (tool, args) executions, used to
        # serve identical calls from cache instead of re-applying a side effect.
        self._idem = IdempotencyStore()

        # wire up sub-agent capability
        for t in self.tools:
            if isinstance(t, AgentTool):
                t._parent_agent = self
            elif isinstance(t, SpawnSubagentTool):
                t._parent_agent = self

    def _full_messages(self) -> list[dict]:
        return [{"role": "system", "content": self._system}] + self.messages

    def _select_tools(self, query: str | None = None) -> list[Tool]:
        if self.tool_selector is not None and query and query.strip():
            return self.tool_selector.select(query, self.tools)
        return self.tools

    def _tool_schemas(self, query: str | None = None) -> list[dict]:
        return [t.schema() for t in self._select_tools(query)]

    def chat(self, user_input: str, on_token=None, on_tool=None) -> str:
        """Process one user message. May involve multiple LLM/tool rounds."""
        self.messages.append({"role": "user", "content": user_input})
        self.context.maybe_compress(self.messages, self.llm)

        for _ in range(self.max_rounds):
            resp = self.llm.chat(
                messages=self._full_messages(),
                tools=self._tool_schemas(user_input),
                on_token=on_token,
            )

            # no tool calls -> LLM is done, return text
            if not resp.tool_calls:
                self.messages.append(resp.message)
                return resp.content

            # tool calls -> execute (parallel when multiple, like Claude Code's
            # StreamingToolExecutor which runs independent tools concurrently)
            self.messages.append(resp.message)

            try:
                if len(resp.tool_calls) == 1:
                    tc = resp.tool_calls[0]
                    if on_tool:
                        on_tool(tc.name, tc.arguments)
                    result = self._exec_tool(tc)
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
                else:
                    # parallel execution for multiple tool calls
                    results = self._exec_tools_parallel(resp.tool_calls, on_tool)
                    for tc, result in zip(resp.tool_calls, results):
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result,
                        })
            except KeyboardInterrupt:
                # Ctrl+C mid-execution would leave the assistant tool_calls
                # message without replies, poisoning the next request; backfill
                self._answer_pending_tool_calls(resp.tool_calls)
                raise

            # compress if tool outputs are big
            self.context.maybe_compress(self.messages, self.llm)

        return "(reached maximum tool-call rounds)"

    def _exec_tool(self, tc) -> str:
        """Execute a single tool call, returning the result string."""
        tool = self._tool_by_name.get(tc.name)
        if tool is None:
            return f"Error: unknown tool '{tc.name}'"
        # Phase 3 planning guard: soft/hard interception at the dispatch layer
        # (see corecoder/planner.py). Soft mode preserves open-ended editing;
        # CORECODER_ENFORCE_PLANNING=1 hard-blocks mutation without a plan.
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
        try:
            # Phase 3 self-correction: deterministic retry strategies on
            # transient failures (retry_same / retry_modified with timeout
            # extension); everything else surfaces for the agent to reflect.
            # Non-idempotent tools pass retry_safe=False so a side effect that
            # already happened before a failure is never double-applied.
            result = run_with_correction(
                tool.execute, retry_safe=idempotent, **tc.arguments
            )
        except Exception as e:
            return f"Error executing {tc.name}: {e}"
        if idempotent:
            self._idem.put(idem_key, result)
        return result

    def _exec_tools_parallel(self, tool_calls, on_tool=None) -> list[str]:
        """Run multiple tool calls concurrently using threads.

        This is inspired by Claude Code's StreamingToolExecutor which starts
        executing tools while the model is still generating.  We simplify to:
        when the model returns N tool calls at once, run them in parallel.
        """
        for tc in tool_calls:
            if on_tool:
                on_tool(tc.name, tc.arguments)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(self._exec_tool, tc) for tc in tool_calls]
            return [f.result() for f in futures]

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

    def reset(self):
        """Clear conversation history."""
        self.messages.clear()
