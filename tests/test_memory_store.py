"""Phase 5 storage layer: CRUD, dual-db, FTS5 manual management, dedup,
sensitive filtering (spec: 8 tests)."""

import time

from mycoder.memory import MemoryEntry, MemoryStore


def _save(store, content, **kw):
    return store.save(MemoryEntry(content=content, **kw))


def test_save_and_get_roundtrip(memory_store):
    mem_id = _save(memory_store, "认证模块使用JWT", type="project")
    entry = memory_store.get(mem_id)
    assert entry is not None
    assert entry.content == "认证模块使用JWT"
    assert entry.type == "project"
    assert entry.scope == "project"
    assert memory_store.project_db_path.exists()


def test_global_scope_persisted_in_global_db(memory_store):
    mem_id = _save(memory_store, "全局约定：代码注释用中文", scope="global")
    entry = memory_store.get(mem_id)
    assert entry.scope == "global"
    assert memory_store.global_db_path.exists()


def test_get_tries_both_dbs(memory_store):
    mem_id = _save(memory_store, "只存在于全局库的内容", scope="global")
    # get() checks the project db first, then the global db
    assert memory_store.get(mem_id) is not None


def test_delete_removes_main_fts_and_vector(memory_store):
    mem_id = _save(memory_store, "待删除的记忆内容")
    assert memory_store.bm25_search("待删除")  # indexed in FTS5
    assert memory_store.delete(mem_id)
    assert memory_store.get(mem_id) is None
    assert memory_store.bm25_search("待删除") == []  # FTS row gone too
    assert mem_id not in [e.id for e in memory_store.list()]
    assert memory_store.delete(mem_id) is False  # idempotent


def test_save_redacts_sensitive_information(memory_store):
    mem_id = _save(
        memory_store,
        "OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz123456 请勿外泄",
    )
    entry = memory_store.get(mem_id)
    assert "sk-abcdefghijklmnopqrstuvwxyz123456" not in entry.content
    assert "[REDACTED" in entry.content


def test_save_dedup_cosine_updates_not_creates(memory_store):
    a = _save(memory_store, "认证模块使用JWT登录")
    b = _save(memory_store, "认证模块使用JWT登录。")
    assert b == a  # near-duplicate (cosine > 0.85) -> same id, updated
    assert len(memory_store.list()) == 1
    entry = memory_store.get(a)
    assert entry.content == "认证模块使用JWT登录。"  # content refreshed

    c = _save(memory_store, "完全不同的主题内容说明")
    assert c != a
    assert len(memory_store.list()) == 2


def test_save_dedup_exact_content_without_embedder(tmp_path):
    store = MemoryStore(
        project_dir=tmp_path / "p", global_dir=tmp_path / "g", embedder=None
    )
    a = _save(store, "完全相同的内容")
    b = _save(store, "完全相同的内容")
    assert b == a
    assert len(store.list()) == 1


def test_confirm_promotes_to_confirmed(memory_store):
    mem_id = _save(memory_store, "需要确认的知识点", confidence=0.5)
    assert memory_store.confirm(mem_id)
    entry = memory_store.get(mem_id)
    assert entry.source == "confirmed"
    assert entry.confidence == 1.0
    assert memory_store.confirm("missing-id") is False


def test_list_order_and_filter(memory_store):
    old = time.time() - 100
    _save(memory_store, "较早的全局记忆", scope="global", created_at=old)
    _save(memory_store, "较新的项目记忆", scope="project")
    newest_first = memory_store.list()
    assert [e.content for e in newest_first][0] == "较新的项目记忆"
    only_project = memory_store.list(scope="project")
    assert [e.content for e in only_project] == ["较新的项目记忆"]
    only_global = memory_store.list(scope="global")
    assert [e.content for e in only_global] == ["较早的全局记忆"]
