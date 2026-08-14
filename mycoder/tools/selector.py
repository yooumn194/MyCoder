"""Dynamic tool selection — inject only the tools relevant to the current task.

When a tool pool grows (built-ins + MCP servers), feeding every tool schema to
the model on every turn wastes tokens and blurs the model's tool choice. The
selector keeps a small core of always-available tools and ranks the rest by
lexical relevance to the user message (word-level, reusing the memory
tokenizer), then injects the top-N.

Disambiguation is a by-product: tools whose name/description match the query
terms surface above the noise, so "搜索文件" pulls grep/list/read to the front
instead of the model guessing among 20 schemas.

Zero dependencies — scoring is pure term overlap on a tokenized bag of words.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from .base import Tool

# Tools always injected regardless of relevance — removing these would break
# basic coding capability even when the query doesn't mention them.
CORE_TOOLS: frozenset[str] = frozenset(
    {
        "read_file",
        "edit_file",
        "write_file",
        "grep_search",
        "list_files",
        "execute_in_sandbox",
        # Planning pair: injecting todo_write without todo_update would let the
        # plan's step never be marked in_progress, and planning_guard then
        # blocks every mutation — a deadlock. Both must always be present.
        "todo_write",
        "todo_update",
        "sync_workspace",   # sandbox sync is required right after execute
        "spawn_subagent",   # delegation is a core loop primitive
        "memory_search",    # agentic RAG core
    }
)


def _tokenize(text: str) -> set[str]:
    try:
        from ..memory.tokenizer import tokenize_chinese

        return set(tokenize_chinese(text).split())
    except Exception:  # pragma: no cover - defensive
        return set((text or "").lower().split())


class ToolSelector:
    """Relevance-ranked tool subset for the current query."""

    def __init__(
        self,
        always_include: set[str] | None = None,
        additional_include: set[str] | None = None,
        top_n: int = 12,
        reserve: int = 4,
    ) -> None:
        """``additional_include`` merges into always_include (e.g. the MCP tools
        the operator explicitly enabled — those must never be dropped by
        ranking). ``reserve`` guarantees the relevance-ranked remainder always
        has at least that many slots, even when the core set grows."""
        base = set(always_include) if always_include else set(CORE_TOOLS)
        base.update(additional_include or ())
        self.always_include = base
        self._reserve = max(0, int(reserve))
        self.top_n = max(len(self.always_include) + self._reserve, top_n)

    def select(self, query: str, tools: Sequence[Tool]) -> list[Tool]:
        """Return the tools to inject: core tools first, then the most
        relevant remainder, capped at top_n. Preserves the given tool order."""
        terms = _tokenize(query)

        def score(tool: Tool) -> float:
            text = f"{tool.name} {tool.description} {json.dumps(tool.parameters or {}, default=str)}"
            toks = _tokenize(text)
            overlap = len(toks & terms)
            name_hit = 1.0 if tool.name in terms else 0.0
            return overlap + name_hit

        core = [t for t in tools if t.name in self.always_include]
        rest = sorted(
            (t for t in tools if t.name not in self.always_include),
            key=score,
            reverse=True,
        )
        return (core + rest)[: self.top_n]
