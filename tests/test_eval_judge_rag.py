"""P2 LLM-as-Judge + RAG eval + document parent-child index."""

from mycoder.memory import HybridRetriever, MemoryStore
from mycoder.memory.document import retrieve_parents, save_document
from eval_bench.judge import LLMJudge
from eval_bench.rag_eval import SAMPLE_DOC, SAMPLE_QUERIES, evaluate


class _Resp:
    def __init__(self, content):
        self.content = content


class _Stub:
    def __init__(self, content):
        self._content = content

    def chat(self, messages, response_format=None):
        return _Resp(self._content)


# ------------------------------------------------------------ LLM-as-Judge
def test_judge_parses_score_and_reasoning():
    judge = LLMJudge(llm=_Stub('{"score": 1.0, "reasoning": "完全正确"}'))
    r = judge.judge("问题", "答案", "参考")
    assert r["score"] == 1.0
    assert "完全正确" in r["reasoning"]


def test_judge_forces_score_to_grid():
    judge = LLMJudge(llm=_Stub('{"score": 0.7, "reasoning": "还行"}'))
    assert judge.judge("q", "a")["score"] == 0.5  # 0.7 -> nearest grid value


def test_judge_falls_back_on_bad_output():
    judge = LLMJudge(llm=_Stub("不是 JSON"))
    r = judge.judge("q", "a")
    assert r["score"] == 0.5  # neutral score, never crashes
    assert isinstance(r["reasoning"], str)


def test_judge_without_llm_returns_neutral():
    assert LLMJudge(llm=None).judge("q", "a")["score"] == 0.5


# ------------------------------------------------------------ judge_run eval
def test_judge_run_reports_distribution():
    """judge_run.evaluate aggregates per-case scores into a quality report."""
    from eval_bench.judge_run import SAMPLE_QA, evaluate

    class _SeqStub:
        """LLMJudge calls self._llm.chat(...); emit one score JSON per call."""

        def __init__(self, scores):
            self._contents = [
                f'{{"score": {s}, "reasoning": "stub"}}' for s in scores
            ]

        def chat(self, messages, response_format=None):
            return _Resp(self._contents.pop(0))

    report = evaluate(SAMPLE_QA, judge=LLMJudge(llm=_SeqStub([1.0, 0.5, 0.0])))
    assert report["total"] == len(SAMPLE_QA)
    assert report["correct"] == 1
    assert report["partial"] == 1
    assert report["wrong"] == 1
    assert report["judge_available"] is True
    # 0.0 < 0.5 default threshold -> one low-score 回流 case
    assert len(report["low_score_cases"]) == 1
    assert report["low_score_cases"][0]["score"] == 0.0


def test_judge_run_without_llm_all_neutral():
    """No judge LLM -> neutral 0.5s and judge_available=False (never crashes)."""
    from eval_bench.judge_run import SAMPLE_QA, evaluate

    report = evaluate(SAMPLE_QA, judge=LLMJudge(llm=None))
    assert report["judge_available"] is False
    assert all(r["score"] == 0.5 for r in report["per_case"])
    assert report["low_score_cases"] == []  # 0.5 not < 0.5


# ------------------------------------------------------------ RAG eval
def test_rag_eval_metrics():
    report = evaluate(SAMPLE_DOC, SAMPLE_QUERIES, k=3)
    assert report["recall@k"] == 1.0  # every query's terms recalled in top-3
    assert report["precision@k"] >= 0.5
    assert report["citation_accuracy"] == 1.0  # top-1 holds the first expected term
    assert len(report["per_query"]) == 3


# ------------------------------------------------- parent-child document index
def test_document_parent_child_and_retrieve_parents(tmp_path):
    store = MemoryStore(
        project_dir=tmp_path / "proj", global_dir=tmp_path / "glob", embedder=None
    )
    doc = (
        "# 认证模块\n"
        "## 登录\n登录流程使用 JWT token 进行身份验证，token 有效期一小时。\n"
        "用户每次登录都会获得新的 token，旧 token 立即失效。\n"
        "## 刷新\n刷新 token 可以签发新的访问令牌，不需要重新登录。\n"
    )
    save_document(store, "d", doc, max_chars=50, overlap_ratio=0.0)

    chunks = store.list_by_metadata("doc_id", "d")
    assert len(chunks) >= 3  # multiple chunks so sections span >1 chunk

    # every child chunk's parent_id points to a real, earlier chunk
    for c in chunks:
        pid = c.metadata.get("parent_id")
        if pid:
            parent = store.get(pid)
            assert parent is not None
            assert parent.metadata["chunk_index"] < c.metadata["chunk_index"]

    # retrieve_parents attaches the larger parent context
    hits = HybridRetriever(store).search("JWT", limit=5)
    enriched = retrieve_parents(store, hits)
    for e in enriched:
        assert "parent_id" in e and "parent_content" in e
        if e.get("parent_id"):
            assert e["parent_content"] is not None
