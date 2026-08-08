"""Orchestration efficiency metrics — the feedback loop for Module A.

Each metric answers a question the multi-agent design must keep honest:
  * delegation_accuracy    did the orchestrator pick the right subagent role?
  * speedup_ratio          did parallelism actually pay off?
  * context_inflation      how much summary cost vs the raw work it replaced?
  * lsp_adoption_rate      is the model reaching for LSP instead of grep?
"""

from dataclasses import dataclass


@dataclass
class OrchestrationMetrics:
    delegation_accuracy: float        # 正确选择 Subagent 的比例
    speedup_ratio: float               # 并行加速比（串行时间/并行时间）
    context_inflation_ratio: float    # Subagent 摘要 Token / 原始对话 Token
    lsp_adoption_rate: float          # LSP 调用次数 / (LSP + grep) 总次数


def delegation_accuracy(correct: int, total: int) -> float:
    """Fraction of subagent choices that were the right role for the task."""
    if total == 0:
        return 1.0
    return correct / total


def speedup_ratio(serial_seconds: float, parallel_seconds: float) -> float:
    """Speedup of parallel over serial; <1 means parallelism didn't pay."""
    if parallel_seconds <= 0:
        return 0.0
    return serial_seconds / parallel_seconds


def context_inflation_ratio(summary_tokens: int, raw_tokens: int) -> float:
    """Summary overhead vs the raw conversation it replaced (low = efficient)."""
    if raw_tokens <= 0:
        return 0.0
    return summary_tokens / raw_tokens


def lsp_adoption_rate(lsp_calls: int, grep_calls: int) -> float:
    """Fraction of symbol lookups that used LSP rather than grep (high = good)."""
    total = lsp_calls + grep_calls
    if total == 0:
        return 0.0
    return lsp_calls / total


def compute(traces: list[dict]) -> OrchestrationMetrics:
    """Compute all metrics from a list of per-task trace dicts."""
    correct = sum(1 for t in traces if t.get("delegation_correct", False))
    serial = sum(t.get("serial_seconds", 0) for t in traces)
    parallel = sum(t.get("parallel_seconds", 0) for t in traces)
    summary_tokens = sum(t.get("summary_tokens", 0) for t in traces)
    raw_tokens = sum(t.get("raw_tokens", 0) for t in traces)
    lsp = sum(t.get("lsp_calls", 0) for t in traces)
    grep = sum(t.get("grep_calls", 0) for t in traces)
    return OrchestrationMetrics(
        delegation_accuracy=delegation_accuracy(correct, len(traces)),
        speedup_ratio=speedup_ratio(serial, parallel),
        context_inflation_ratio=context_inflation_ratio(summary_tokens, raw_tokens),
        lsp_adoption_rate=lsp_adoption_rate(lsp, grep),
    )
