"""Document-level RAG layer: semantic chunking, incremental reindex, and
rule-based rerank (mycoder/memory/document.py + retriever.RuleReranker)."""

from mycoder.memory import MemoryEntry, MemoryStore, HybridRetriever
from mycoder.memory.document import chunk_document, reindex_document, save_document
from mycoder.memory.retriever import Reranker, RuleReranker

MARKDOWN = """\
# Intro
This is the intro paragraph about the auth module.

## Login
The login flow uses JWT tokens. Tokens expire in one hour.

## Refresh
A refresh token can mint a new access token.
"""

PYTHON = """\
def add(a, b):
    return a + b


def multiply(a, b):
    return a * b


class Calculator:
    def run(self, op, a, b):
        return op(a, b)
"""


def _store(tmp_path, embedder=None):
    return MemoryStore(
        project_dir=tmp_path / "proj", global_dir=tmp_path / "glob", embedder=embedder
    )


# ------------------------------------------------------------------ chunking
def test_chunk_markdown_headings():
    chunks = chunk_document(MARKDOWN, max_chars=120)
    assert len(chunks) >= 2
    assert chunks[0].text.lstrip().startswith("# Intro")
    assert any(c.heading and "Login" in c.heading for c in chunks)
    assert chunks[0].start_line >= 1


def test_chunk_python_def_boundaries():
    # overlap off so we test boundary detection in isolation
    chunks = chunk_document(PYTHON, max_chars=100, overlap_ratio=0)
    texts = "\n".join(c.text for c in chunks)
    for symbol in ("def add", "def multiply", "class Calculator"):
        assert symbol in texts
        # each definition survives intact at a line start (block boundary
        # respected — never split mid-line by packing)
        assert any(
            line.lstrip().startswith(symbol)
            for c in chunks
            for line in c.text.split("\n")
        )
    assert all(len(c.text) <= 100 + int(100 * 0.15) + 5 for c in chunks)


def test_chunk_overlap_between_adjacent():
    text = ("para " * 40) + "\n\n" + ("second " * 40) + "\n\n" + ("third " * 40)
    chunks = chunk_document(text, max_chars=120, overlap_ratio=0.2)
    assert len(chunks) >= 2
    # the overlap appears at the HEAD of the next chunk (a tail of the previous)
    head = chunks[1].text.split("\n")[0]
    assert head and head in chunks[0].text


def test_chunk_oversized_single_paragraph():
    big = "word " * 3000  # ~15000 chars, no semantic boundaries
    chunks = chunk_document(big, max_chars=500)
    assert len(chunks) > 5
    budget = 500 + int(500 * 0.15) + 10
    assert all(len(c.text) <= budget for c in chunks)
    # content preserved across chunks
    joined = " ".join(c.text for c in chunks)
    assert joined.count("word") >= 2990


def test_chunk_empty():
    assert chunk_document("") == []
    assert chunk_document("   \n  ") == []


# --------------------------------------------------------- save / reindex
def test_save_document_creates_chunks(tmp_path):
    store = _store(tmp_path, embedder=None)
    n = save_document(store, "readme", MARKDOWN, max_chars=120)
    chunks = store.list_by_metadata("doc_id", "readme")
    assert n == len(chunks) >= 3
    for c in chunks:
        assert c.metadata["chunk_index"] == int(c.id.rsplit(":", 1)[1])
        assert c.metadata["chunk_hash"]
    # deterministic ids, no duplicates
    assert len({c.id for c in chunks}) == len(chunks)


def test_save_document_upsert_keeps_ids(tmp_path):
    store = _store(tmp_path, embedder=None)
    save_document(store, "doc", MARKDOWN, max_chars=120)
    first_ids = {e.id for e in store.list_by_metadata("doc_id", "doc")}
    save_document(store, "doc", MARKDOWN, max_chars=120)  # same content -> upsert, no dup
    second_ids = {e.id for e in store.list_by_metadata("doc_id", "doc")}
    assert first_ids == second_ids
    assert len(second_ids) == len(first_ids)


def test_reindex_unchanged_skips(tmp_path):
    store = _store(tmp_path, embedder=None)
    save_document(store, "readme", MARKDOWN, max_chars=120)
    diff = reindex_document(store, "readme", MARKDOWN, max_chars=120)
    assert diff["added"] == 0 and diff["updated"] == 0 and diff["removed"] == 0
    assert diff["unchanged"] > 0


def test_reindex_added_and_removed(tmp_path):
    store = _store(tmp_path, embedder=None)
    save_document(store, "readme", MARKDOWN, max_chars=120)
    before = len(store.list_by_metadata("doc_id", "readme"))

    modified = MARKDOWN + "\n\n## Caching\n" + ("cache layer note\n" * 20)
    diff = reindex_document(store, "readme", modified, max_chars=120)
    assert diff["added"] >= 1
    after = len(store.list_by_metadata("doc_id", "readme"))
    assert after == before + diff["added"] - diff["removed"]

    shrink = reindex_document(store, "readme", MARKDOWN[:60], max_chars=120)
    assert shrink["removed"] >= 1
    assert shrink["added"] == 0


def test_chunks_retrievable_by_hybrid(tmp_path):
    store = _store(tmp_path, embedder=None)
    save_document(store, "readme", MARKDOWN, max_chars=80)
    hits = HybridRetriever(store).search("JWT", limit=5)
    assert any("JWT" in h["content"] for h in hits)
    assert any(h.get("metadata", {}).get("doc_id") == "readme" for h in hits)


# ------------------------------------------------------------ rerank
def test_rerank_orders_by_query_hits():
    # realistic RRF-scale scores: a ranks higher (0.033 > 0.017) but matches
    # fewer query terms; the reranker should promote b.
    r = RuleReranker(ideal_min=0)  # no length penalty on tiny test strings
    results = [
        {"id": "a", "content": "JWT 令牌过期策略说明", "score": 0.0333},
        {"id": "b", "content": "JWT 令牌 刷新 JWT 续期 策略说明", "score": 0.0172},
    ]
    out = r.rerank("JWT 令牌 刷新", results)
    assert out[0]["id"] == "b"  # more query terms present beats higher RRF score


def test_rerank_length_penalty():
    r = RuleReranker(ideal_min=120, ideal_max=1600)
    results = [
        {"id": "short", "content": "JWT", "score": 0.8},
        {"id": "normal", "content": "JWT " + "策略说明 " * 40, "score": 0.79},
    ]
    out = r.rerank("JWT 策略", results)
    assert out[0]["id"] == "normal"  # the 3-char chunk is penalized below normal


def test_retriever_applies_injected_reranker(tmp_path):
    class _ReverseReranker(Reranker):
        def rerank(self, query, results):
            return list(reversed(results))

    store = _store(tmp_path, embedder=None)
    store.save(MemoryEntry(content="AAA 内容一", type="fact"))
    store.save(MemoryEntry(content="BBB 内容二", type="fact"))
    rev = HybridRetriever(store, reranker=_ReverseReranker())
    plain = HybridRetriever(store)
    hits = rev.search("内容", limit=2, rerank=True)
    base = plain.search("内容", limit=2, rerank=False)
    assert len(hits) == 2 and len(base) == 2
    assert hits[0]["id"] == base[-1]["id"]  # pipeline reversed the order


def test_store_save_dedup_false_keeps_ids(tmp_path):
    store = _store(tmp_path, embedder=None)
    a = store.save(MemoryEntry(id="x1", content="认证模块使用JWT登录", type="fact"), dedup=False)
    b = store.save(MemoryEntry(id="x2", content="认证模块使用JWT登录", type="fact"), dedup=False)
    assert a == "x1" and b == "x2"
    assert store.get("x1") is not None and store.get("x2") is not None
    # default dedup still merges a fresh save into a near-duplicate
    merged = store.save(MemoryEntry(content="认证模块使用JWT登录", type="fact"))
    assert merged in ("x1", "x2")
    assert len(store.list()) == 2
