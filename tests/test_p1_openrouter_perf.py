"""Offline tests for the P1-4/P1-5 OpenRouter benchmark harness."""

import pytest

from eval_bench.p1_openrouter_perf import (
    _safe_error,
    build_openrouter_llm,
    main,
    measure_chat,
    render_markdown,
    run_memory_benchmark,
    run_reasoning_benchmark,
)
from mycoder.llm import LLMResponse


class _FakeLLM:
    def chat(self, messages, on_token=None):
        system = messages[0]["content"]
        user = messages[-1]["content"]
        if "Reasoning mode: ReAct" in system:
            content = "Thought: 读取配置并定位 OpenRouter provider 的 fast 设置，最后报告结果。"
        elif "Plan-and-Execute" in system:
            content = "Plan: 1. 划分模块边界 2. 按依赖顺序迁移 3. 灰度并准备回滚 4. 测试验证。"
        elif "Reasoning mode: Reflection" in system:
            content = "Reflection: 先复现重复写，再用幂等键处理部分流失败，最后补回归测试。"
        elif "JWT" in user:
            content = "120 分钟" if "120 分钟" in user else "15 分钟"
        elif "PostgreSQL" in user:
            content = "3306" if "3306" in user else "5432"
        else:
            content = "3.9" if "3.9" in user else "3.12"
        if on_token:
            on_token(content[:1])
        return LLMResponse(
            content=content,
            prompt_tokens=20,
            completion_tokens=10,
            reasoning_tokens=2,
        )


def test_measure_chat_captures_stream_and_usage():
    result = measure_chat(
        _FakeLLM(),
        [{"role": "system", "content": "Reasoning mode: ReAct"}, {"role": "user", "content": "x"}],
    )
    assert result["status"] == "success"
    assert result["ttft_ms"] is not None
    assert result["reasoning_tokens"] == 2


def test_reasoning_benchmark_compares_all_strategies():
    report = run_reasoning_benchmark(_FakeLLM(), repeats=1)
    assert report["call_count"] == 9
    assert report["successful_calls"] == 9
    assert set(report["by_strategy"]) == {"react", "plan_execute", "reflection"}
    assert report["by_strategy"]["plan_execute"]["marker_adherence_rate"] == 1.0


def test_memory_benchmark_measures_before_after(tmp_path):
    report = run_memory_benchmark(_FakeLLM(), base_dir=tmp_path)
    assert report["call_count"] == 6
    assert report["conflict_pair_recall"] == 1.0
    assert report["before"]["answer_accuracy"] == 0.0
    assert report["after"]["answer_accuracy"] == 1.0
    assert report["accuracy_delta"] == 1.0


def test_openrouter_key_is_required(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        build_openrouter_llm()


def test_markdown_and_dry_run(capsys):
    report = {
        "generated_at": "now",
        "model": "m",
        "p1_5_memory": {
            "case_count": 1,
            "call_count": 2,
            "conflict_pair_precision": 1.0,
            "conflict_pair_recall": 1.0,
            "before": {"answer_accuracy": 0.0},
            "after": {"answer_accuracy": 1.0},
            "accuracy_delta": 1.0,
            "avg_audit_latency_ms": 1.0,
            "avg_correction_latency_ms": 1.0,
        },
    }
    assert "Accuracy delta" in render_markdown(report)
    assert main(["--dry-run", "--suite", "all"]) == 0
    assert "planned_openrouter_calls=15" in capsys.readouterr().out


def test_markdown_handles_failed_strategy_calls():
    report = {
        "generated_at": "now",
        "model": "m",
        "p1_4_reasoning": {
            "case_count": 1,
            "repeats": 1,
            "call_count": 1,
            "successful_calls": 0,
            "by_strategy": {
                "react": {
                    "success_rate": 0.0,
                    "avg_latency_ms": None,
                    "p95_latency_ms": None,
                    "avg_ttft_ms": None,
                    "avg_completion_tokens": None,
                    "requirement_coverage": None,
                    "marker_adherence_rate": 0.0,
                }
            }
        },
    }
    assert "n/a" in render_markdown(report)


def test_provider_error_is_sanitized():
    class _ProviderError(Exception):
        status_code = 402
        body = {"error": {"message": "Provider returned error"}}

        def __str__(self):
            return "failed for user_secretIdentifier"

    safe = _safe_error(_ProviderError())
    assert safe == "HTTP 402: Provider returned error"
    assert "user_" not in safe
