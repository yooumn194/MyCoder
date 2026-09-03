"""One production construction path for CLI, API and sub-agents."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from .agent import Agent
from .tools.base import Tool
from .tools.selector import ToolSelector

log = logging.getLogger(__name__)


class AgentFactory:
    """Build Agents with the same selection, memory and budget capabilities.

    Runtime entry points may still restrict the tool list, model, context or
    reasoning strategy, but capability wiring lives here instead of drifting
    independently across CLI, HTTP and SubagentRunner.
    """

    def __init__(
        self,
        *,
        llm: Any,
        tools: Sequence[Tool],
        max_context_tokens: int = 128_000,
        memory=None,
        tool_selector=None,
        memory_compressor=None,
        experience_store=None,
        budget_guard=None,
        memory_store=None,
    ) -> None:
        self.llm = llm
        self.tools = list(tools)
        self.max_context_tokens = max_context_tokens
        self.memory = memory
        self.tool_selector = tool_selector
        self.memory_compressor = memory_compressor
        self.experience_store = experience_store
        self.budget_guard = budget_guard
        self.memory_store = memory_store

    @classmethod
    def from_defaults(
        cls,
        *,
        llm: Any,
        tools: Sequence[Tool],
        max_context_tokens: int = 128_000,
        budget_guard=None,
        additional_tool_names: set[str] | None = None,
        enable_memory: bool = True,
    ) -> "AgentFactory":
        memory = None
        memory_compressor = None
        memory_store = None
        experience_store = None
        if enable_memory:
            try:
                from .memory.compressor import MemoryCompressor
                from .memory.config import load_memory_config
                from .memory.experience import remember_replan
                from .memory.integration import MemoryIntegration
                from .memory.query_rewrite import LLMQueryRewriter
                from .memory.retriever import HybridRetriever
                from .memory.store import get_store

                mem_cfg = load_memory_config()["memory"]
                memory_store = get_store(mem_cfg.get("embedder"))
                memory_store.filter_sensitive = bool(
                    mem_cfg.get("filter_sensitive", True)
                )
                retriever = HybridRetriever(
                    memory_store,
                    rrf_k=int(mem_cfg.get("rrf_k", 60)),
                    query_rewriter=LLMQueryRewriter(llm),
                )
                memory = MemoryIntegration(
                    store=memory_store,
                    retriever=retriever,
                    max_tokens=int(mem_cfg.get("max_tokens", 2048)),
                ).install()
                memory_compressor = MemoryCompressor(memory_store, llm)
                experience_store = remember_replan
            except Exception as exc:  # noqa: BLE001 - memory is optional
                log.warning("agent memory wiring disabled: %s", exc)

        return cls(
            llm=llm,
            tools=tools,
            max_context_tokens=max_context_tokens,
            memory=memory,
            tool_selector=ToolSelector(
                additional_include=set(additional_tool_names or ())
            ),
            memory_compressor=memory_compressor,
            experience_store=experience_store,
            budget_guard=budget_guard,
            memory_store=memory_store,
        )

    def build(
        self,
        *,
        llm=None,
        tools: Sequence[Tool] | None = None,
        max_context_tokens: int | None = None,
        max_rounds: int = 50,
        reasoning_strategy: str | None = None,
        budget_guard=None,
    ) -> Agent:
        return Agent(
            llm=llm if llm is not None else self.llm,
            tools=list(tools) if tools is not None else list(self.tools),
            max_context_tokens=(
                max_context_tokens
                if max_context_tokens is not None
                else self.max_context_tokens
            ),
            max_rounds=max_rounds,
            memory=self.memory,
            tool_selector=self.tool_selector,
            reasoning_strategy=reasoning_strategy,
            memory_compressor=self.memory_compressor,
            experience_store=self.experience_store,
            budget_guard=budget_guard if budget_guard is not None else self.budget_guard,
        )

    def maintain_memory(self, *, close: bool = False) -> None:
        """Run CLI-style maintenance; API callers keep the shared store open."""
        if self.memory_store is None:
            return
        from .memory.compressor import MemoryCompressor
        from .memory.maintenance import MemoryMaintainer

        maintainer = MemoryMaintainer(self.memory_store)
        maintainer.decay()
        maintainer.compact()
        MemoryCompressor(self.memory_store, self.llm).summarize_cluster(
            scope="project", min_count=20
        )
        if close:
            self.memory_store.close()
