"""Phase 5 memory tools: save / search / list / forget / confirm / stats.

Every tool lazily resolves a MemoryStore (default: the process singleton
created under ~/.mycoder) so importing the registry has zero side effects;
tests inject a store built on tmp_path.
"""

from __future__ import annotations

from ..memory.retriever import HybridRetriever
from ..memory.store import MemoryStore, get_store
from ..memory.types import MEMORY_SCOPES, MEMORY_TYPES, MemoryEntry
from .base import Tool


class MemorySaveTool(Tool):
    name = "memory_save"
    description = (
        "Persist a cross-session memory. content is redacted for secrets and "
        "deduplicated (a near-duplicate is updated, not duplicated). type is "
        f"one of {', '.join(MEMORY_TYPES)}; scope is project (this repo) or global."
    )
    parameters = {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "What to remember"},
            "type": {
                "type": "string",
                "enum": list(MEMORY_TYPES),
                "description": "Memory kind",
            },
            "scope": {
                "type": "string",
                "enum": list(MEMORY_SCOPES),
                "description": "project = repo-local, global = ~/.mycoder",
            },
        },
        "required": ["content"],
    }

    def __init__(self, *, store: MemoryStore | None = None) -> None:
        self._store_ref = store

    def _store(self) -> MemoryStore:
        return self._store_ref or get_store()

    def execute(self, content: str, type: str = "fact", scope: str = "project") -> str:
        entry = MemoryEntry(content=content, type=type, scope=scope, source="user")
        mem_id = self._store().save(entry)
        return f"✅ 已保存记忆 [{mem_id[:8]}]（{type}/{scope}）"


class MemorySearchTool(Tool):
    name = "memory_search"
    description = (
        "Hybrid cross-session memory search (BM25 + vector, RRF-fused). "
        "Returns relevant memories with their type/scope/confidence."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to look for"},
            "scope": {
                "type": "string",
                "enum": list(MEMORY_SCOPES),
                "description": "Restrict to a db (default: both)",
            },
            "limit": {"type": "integer", "description": "Max results", "default": 10},
        },
        "required": ["query"],
    }

    def __init__(
        self,
        *,
        store: MemoryStore | None = None,
        retriever: HybridRetriever | None = None,
        agentic: bool = True,
    ) -> None:
        self._store_ref = store
        self._retriever_ref = retriever
        # P1 agentic RAG: when the default retriever is built (no explicit
        # `retriever` injected), wrap it in the retrieve->judge->re-query loop
        # so a thin first pass keeps searching instead of giving up.
        self.agentic = agentic

    def _retriever(self):
        if self._retriever_ref is not None:
            return self._retriever_ref
        base = HybridRetriever(self._store_ref or get_store())
        if not self.agentic:
            return base
        from ..memory.agentic import AgenticRetriever

        return AgenticRetriever(base)

    def execute(self, query: str, scope: str | None = None, limit: int = 10) -> str:
        results = self._retriever().search(query, limit=limit, scope=scope)
        if not results:
            return f"🔍 未找到相关记忆（query: {query}）"
        # P2 citation tracing: results are numbered so the agent can cite [n];
        # the source anchor (doc location) makes the citation traceable.
        lines = [
            f"[{i + 1}] ({_source_anchor(r)}|{r['type']}|conf={r['confidence']:.2f}) {r['content']}"
            for i, r in enumerate(results)
        ]
        return (
            "🔍 相关记忆（回答时用 [n] 标注来源编号；检索不足时明说「未找到相关信息」）:\n"
            + "\n".join(lines)
        )


def _source_anchor(r: dict) -> str:
    """A cite-able source anchor for a retrieved memory: the doc location when
    the entry came from a chunked document (P2 citation tracing), else the id."""
    md = r.get("metadata") or {}
    doc_id = md.get("doc_id")
    if doc_id:
        anchor = f"doc:{str(doc_id)[:12]}"
        start = md.get("start_line")
        if start is not None:
            anchor += f":{start}"
        return anchor
    return str(r["id"])[:8]


class MemoryListTool(Tool):
    name = "memory_list"
    description = "List current memories (newest first)."
    parameters = {
        "type": "object",
        "properties": {
            "type": {
                "type": "string",
                "enum": list(MEMORY_TYPES),
                "description": "Filter by memory type",
            },
            "scope": {
                "type": "string",
                "enum": list(MEMORY_SCOPES),
                "description": "Filter by scope (default: both)",
            },
        },
        "required": [],
    }

    def __init__(self, *, store: MemoryStore | None = None) -> None:
        self._store_ref = store

    def _store(self) -> MemoryStore:
        return self._store_ref or get_store()

    def execute(self, type: str | None = None, scope: str | None = None) -> str:
        entries = self._store().list(scope=scope, type=type)
        if not entries:
            return "📭 当前没有记忆"
        lines = [
            f"- [{e.id[:8]}|{e.type}|{e.scope}|{e.source}|conf={e.confidence:.2f}] {e.content}"
            for e in entries
        ]
        return f"📚 共 {len(entries)} 条记忆:\n" + "\n".join(lines)


class MemoryForgetTool(Tool):
    name = "memory_forget"
    description = "Delete a memory by id."
    parameters = {
        "type": "object",
        "properties": {"memory_id": {"type": "string", "description": "Memory id"}},
        "required": ["memory_id"],
    }

    def __init__(self, *, store: MemoryStore | None = None) -> None:
        self._store_ref = store

    def _store(self) -> MemoryStore:
        return self._store_ref or get_store()

    def execute(self, memory_id: str) -> str:
        if self._store().delete(memory_id):
            return f"🗑️ 已删除记忆 [{memory_id[:8]}]"
        return f"❌ 未找到记忆 [{memory_id[:8]}]"


class MemoryConfirmTool(Tool):
    name = "memory_confirm"
    description = (
        "Promote a memory to confirmed (confidence 1.0, never decays). Use for "
        "memories the user explicitly validated."
    )
    parameters = {
        "type": "object",
        "properties": {"memory_id": {"type": "string", "description": "Memory id"}},
        "required": ["memory_id"],
    }

    def __init__(self, *, store: MemoryStore | None = None) -> None:
        self._store_ref = store

    def _store(self) -> MemoryStore:
        return self._store_ref or get_store()

    def execute(self, memory_id: str) -> str:
        if self._store().confirm(memory_id):
            return f"⭐ 记忆 [{memory_id[:8]}] 已确认为 confirmed（置信度 1.0，永不衰减）"
        return f"❌ 未找到记忆 [{memory_id[:8]}]"


class MemoryCorrectTool(Tool):
    name = "memory_correct"
    description = (
        "Fix a polluted/wrong memory: provide content to replace it (clears "
        "deprecation and raises confidence), or omit content to deprecate it so "
        "it stops surfacing. The reverse of memory_confirm."
    )
    parameters = {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "Memory id to correct"},
            "content": {
                "type": "string",
                "description": "Optional corrected content (replaces the wrong one)",
            },
            "reason": {
                "type": "string",
                "description": "Why it was corrected (stored as deprecated_by when no content)",
                "default": "corrected",
            },
        },
        "required": ["memory_id"],
    }

    def __init__(self, *, store: MemoryStore | None = None) -> None:
        self._store_ref = store

    def _store(self) -> MemoryStore:
        return self._store_ref or get_store()

    def execute(self, memory_id: str, content: str | None = None, reason: str = "corrected") -> str:
        from ..memory.maintenance import MemoryMaintainer

        if MemoryMaintainer(self._store()).correct_memory(
            memory_id, content=content, reason=reason
        ):
            if content is not None:
                return f"✅ 记忆 [{memory_id[:8]}] 已用新内容纠正（置信度已提升）"
            return f"🗑️ 记忆 [{memory_id[:8]}] 已废弃（不再参与检索）"
        return f"❌ 未找到记忆 [{memory_id[:8]}]"


class MemoryStatsTool(Tool):
    name = "memory_stats"
    description = "Show memory store statistics (counts, type distribution, avg confidence)."
    parameters = {"type": "object", "properties": {}, "required": []}

    def __init__(self, *, store: MemoryStore | None = None) -> None:
        self._store_ref = store

    def _store(self) -> MemoryStore:
        return self._store_ref or get_store()

    def execute(self) -> str:
        from ..memory.maintenance import MemoryMaintainer

        stats = MemoryMaintainer(self._store()).get_stats()
        types = ", ".join(f"{k}:{v}" for k, v in stats["by_type"].items()) or "—"
        return (
            f"📊 记忆统计\n"
            f"  总数: {stats['total']}  活跃: {stats['active']}  已废弃: {stats['deprecated']}\n"
            f"  平均置信度: {stats['avg_confidence']}\n"
            f"  类型分布: {types}\n"
            f"  库分布: project:{stats['by_scope'].get('project', 0)} global:{stats['by_scope'].get('global', 0)}"
        )
