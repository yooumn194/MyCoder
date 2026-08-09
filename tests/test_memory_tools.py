"""Phase 5 memory tools: save / search / list / forget / confirm / stats
(spec: 7 tests)."""

import time

from corecoder.memory import MemoryEntry, HybridRetriever
from corecoder.tools.memory_tools import (
    MemoryConfirmTool,
    MemoryForgetTool,
    MemoryListTool,
    MemorySaveTool,
    MemorySearchTool,
    MemoryStatsTool,
)


def test_memory_save_creates_memory(memory_store):
    tool = MemorySaveTool(store=memory_store)
    out = tool.execute(content="记住：构建用 make", type="fact", scope="project")
    assert "已保存" in out
    assert len(memory_store.list()) == 1


def test_memory_search_returns_formatted_results(memory_store):
    MemorySaveTool(store=memory_store).execute(content="认证模块使用JWT")
    tool = MemorySearchTool(store=memory_store, retriever=HybridRetriever(memory_store))
    out = tool.execute(query="认证模块")
    assert "相关记忆" in out
    assert "认证模块使用JWT" in out
    assert "|fact|project|conf=" in out  # type/scope/confidence rendered


def test_memory_list_newest_first(memory_store):
    old = time.time() - 100
    memory_store.save(MemoryEntry(content="较早的记忆", created_at=old))
    memory_store.save(MemoryEntry(content="较新的记忆", created_at=time.time()))
    out = MemoryListTool(store=memory_store).execute()
    assert "共 2 条记忆" in out
    assert out.index("较新的记忆") < out.index("较早的记忆")


def test_memory_list_filters_by_type_and_scope(memory_store):
    MemorySaveTool(store=memory_store).execute(content="一条 user 记忆", type="user")
    MemorySaveTool(store=memory_store).execute(content="一条 global 记忆", scope="global")
    out = MemoryListTool(store=memory_store).execute(type="user")
    assert "一条 user 记忆" in out and "一条 global 记忆" not in out


def test_memory_forget_deletes(memory_store):
    MemorySaveTool(store=memory_store).execute(content="待遗忘的记忆")
    mem_id = memory_store.list()[0].id
    out = MemoryForgetTool(store=memory_store).execute(memory_id=mem_id)
    assert "已删除" in out
    assert memory_store.get(mem_id) is None
    assert "未找到" in MemoryForgetTool(store=memory_store).execute(memory_id="missing")


def test_memory_confirm_promotes(memory_store):
    MemorySaveTool(store=memory_store).execute(content="确认这条记忆")
    mem_id = memory_store.list()[0].id
    out = MemoryConfirmTool(store=memory_store).execute(memory_id=mem_id)
    assert "confirmed" in out
    entry = memory_store.get(mem_id)
    assert entry.source == "confirmed"
    assert entry.confidence == 1.0


def test_memory_stats_reports_counts(memory_store):
    MemorySaveTool(store=memory_store).execute(content="记忆一")
    MemorySaveTool(store=memory_store).execute(content="记忆二", scope="global")
    out = MemoryStatsTool(store=memory_store).execute()
    assert "总数: 2" in out
    assert "活跃: 2" in out
    assert "project:1 global:1" in out


def test_memory_save_dedups_through_the_tool(memory_store):
    tool = MemorySaveTool(store=memory_store)
    tool.execute(content="认证模块使用JWT登录")
    tool.execute(content="认证模块使用JWT登录。")
    assert len(memory_store.list()) == 1
