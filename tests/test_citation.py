"""P2 citation tracing + badcase 回流 tests.

Covers mycoder/memory/citation.py, the rag_eval citation-integrity dimension,
the memory-search source anchor, the judge_run badcase queue/regenerate loop,
and the failure-kb few-shot prevention.
"""

import json

from mycoder.eval.failure_kb import FailureKnowledgeBase, FailurePattern
from mycoder.memory.citation import (
    extract_citation_ids,
    grounded_answer_prompt,
    verify_citations,
)
from mycoder.tools.memory_tools import _source_anchor
from eval_bench.judge_run import _append_queue, _regenerate
from eval_bench.rag_eval import SAMPLE_ANSWERS, evaluate_citations


# ------------------------------------------------------------- P2-① citation
def test_extract_citation_ids():
    assert extract_citation_ids("[1] 是 A，[2, 3] 是 B。") == [1, 2, 3]
    assert extract_citation_ids("没有引用的答案") == []
    assert extract_citation_ids(None) == []


def test_verify_citations_valid_and_hallucinated():
    v = verify_citations("登录用 JWT[1]，退款[2]。", 2)
    assert v["valid"] is True
    assert v["missing"] == []

    h = verify_citations("登录用 JWT[9]。", 2)
    assert h["valid"] is False
    assert h["missing"] == [9]  # 幻觉引用：越界下标

    u = verify_citations("登录用 JWT。", 2)
    assert u["valid"] is False  # 事实陈述但无引用
    assert u["citation_density"] == 0.0


def test_grounded_answer_prompt_numbered():
    p = grounded_answer_prompt(["ctxA", "ctxB"])
    assert "[1] ctxA" in p and "[2] ctxB" in p
    assert "未找到相关信息" in p  # no-evidence fallback demanded


def test_rag_eval_citation_integrity():
    report = evaluate_citations(SAMPLE_ANSWERS)
    assert report["citation_integrity"] < 1.0  # 含幻觉 / 无引用样例
    assert report["hallucinated_reference_ratio"] > 0.0
    by_id = {v["id"]: v for v in report["per_answer"]}
    assert by_id["good"]["valid"] is True
    assert by_id["hallucinated"]["missing"] == [9]
    assert by_id["uncited"]["valid"] is False


def test_source_anchor_from_doc_metadata():
    r = {"id": "abc123", "metadata": {"doc_id": "doc-xyz", "start_line": 42}}
    assert _source_anchor(r) == "doc:doc-xyz:42"
    assert _source_anchor({"id": "abc123", "metadata": {}}) == "abc123"


# ------------------------------------------------------------- P2-② 回流
def test_append_queue_dedups(tmp_path):
    q = tmp_path / "badcase_queue.jsonl"
    low = [
        {"id": "a", "score": 0.0, "reasoning": "x"},
        {"id": "b", "score": 0.0, "reasoning": "y"},
    ]
    assert _append_queue(q, low) == 2
    assert _append_queue(q, [{"id": "a", "score": 0.0}]) == 0  # deduped
    lines = q.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2


def test_regenerate_emits_manifest(tmp_path):
    q = tmp_path / "badcase_queue.jsonl"
    q.write_text(json.dumps({"id": "a", "score": 0.0, "reasoning": "r"}) + "\n", encoding="utf-8")
    out = tmp_path / "regenerate_manifest.json"
    n = _regenerate(q, out)
    assert n == 1
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["sample_count"] == 1
    assert data["samples"][0]["id"] == "a"
    assert "review_instruction" in data  # 人工审核 → 入数据集 的桥


def test_judge_run_case_with_contexts_gets_citation_verdict():
    from eval_bench.judge_run import evaluate

    class _NeutralJudge:
        _llm = None

        def judge(self, question, answer, reference=None):
            return {"score": 0.5, "reasoning": "neutral"}

    report = evaluate(
        [
            {
                "id": "c1",
                "question": "q",
                "answer": "登录用 JWT[1]",
                "contexts": ["JWT 登录流程", "退款流程"],
            }
        ],
        judge=_NeutralJudge(),
    )
    assert "citation" in report["per_case"][0]
    assert report["per_case"][0]["citation"]["valid"] is True


# ------------------------------------------------- failure-kb few-shot 预防
def test_failure_kb_few_shots_empty_and_populated():
    kb = FailureKnowledgeBase()
    assert kb.few_shots() == ""
    kb.record_failure("case1", FailurePattern.TOOL_SELECTION, {})
    kb.record_failure("case2", FailurePattern.TOOL_SELECTION, {})
    shots = kb.few_shots()
    assert "tool_selection" in shots
    assert "2 次" in shots
    assert "对策" in shots


def test_failure_kb_inject_few_shots():
    kb = FailureKnowledgeBase()
    assert kb.inject_few_shots("base") == "base"  # 无记录时 no-op
    kb.record_failure("c", FailurePattern.CONTEXT_LOSS, {})
    prompt = kb.inject_few_shots("base prompt")
    assert prompt.startswith("base prompt")
    assert "失败经验" in prompt
    assert "context_loss" in prompt
