"""Phase 5 E2E: write -> retrieve -> inject -> influence output (spec: 3)."""

import pytest

import corecoder.planner as planner_mod
from corecoder.memory import MemoryEntry, HybridRetriever, MemoryStore
from corecoder.memory.embedder import HashingEmbedder
from corecoder.memory.integration import MemoryIntegration
from corecoder.memory.prompt import MemoryPromptInjector
from corecoder.planner import PlanStore
from corecoder.tools.correction import run_with_correction
from corecoder.tools.todo_tools import TodoWriteTool


@pytest.fixture
def e2e_store(tmp_path):
    return MemoryStore(
        project_dir=tmp_path / "proj",
        global_dir=tmp_path / "glob",
        embedder=HashingEmbedder(),
    )


def test_e2e_write_search_inject_influence(tmp_path, e2e_store):
    retriever = HybridRetriever(e2e_store)
    MemoryIntegration(store=e2e_store, retriever=retriever).install()

    # 1) write
    mem_id = e2e_store.save(
        MemoryEntry(content="用户偏好：代码注释必须使用中文", type="user")
    )
    # 2) retrieve
    hits = retriever.search("代码注释", limit=5)
    assert any(h["id"] == mem_id for h in hits)
    # 3) inject
    section = MemoryPromptInjector(retriever).build_memory_section("代码注释")
    assert "代码注释必须使用中文" in section
    # 4) influence output
    assert planner_mod.planning_guard("todo_write", query="代码注释") is None
    tool = TodoWriteTool(store=PlanStore(base_dir=tmp_path / "plans"))
    out = tool.execute(
        task_goal="添加注释规范",
        todos=[{"id": "s1", "description": "编写注释规范"}],
    )
    assert "代码注释必须使用中文" in out


def test_e2e_cross_session_global_persistence(tmp_path):
    paths = (tmp_path / "proj", tmp_path / "glob")

    # session 1 writes a global memory
    store1 = MemoryStore(
        project_dir=paths[0], global_dir=paths[1], embedder=HashingEmbedder()
    )
    store1.save(
        MemoryEntry(content="全局规范：接口错误码统一为 2xx/4xx", scope="global")
    )
    store1.close()

    # session 2 (fresh store over the same db files) can retrieve it
    store2 = MemoryStore(
        project_dir=paths[0], global_dir=paths[1], embedder=HashingEmbedder()
    )
    hits = HybridRetriever(store2).search("错误码", limit=5)
    assert any("接口错误码" in h["content"] for h in hits)
    assert hits[0]["scope"] == "global"


def test_e2e_pattern_settlement_feeds_retrieval(tmp_path, e2e_store):
    retriever = HybridRetriever(e2e_store)
    MemoryIntegration(store=e2e_store, retriever=retriever).install()

    calls = {"n": 0}

    def _flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise TimeoutError("网络抖动")
        return "ok"

    assert run_with_correction(_flaky, sleep_fn=lambda _: None) == "ok"

    # the settled PATTERN memory is now retrievable and injectable
    patterns = e2e_store.list(type="pattern")
    assert len(patterns) == 1
    section = MemoryPromptInjector(retriever).build_memory_section("重试")
    assert "重试" in section
    assert patterns[0].id in {h["id"] for h in retriever.search("重试", limit=5)}
