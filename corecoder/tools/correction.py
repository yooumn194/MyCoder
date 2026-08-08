"""Self-correction: deterministic error classification -> strategy mapping.

Optimization point #2: instead of a one-size-fits-all "small model diagnoses
the failure", classify the error with a deterministic rule table (zero tokens)
and map it onto a concrete strategy — retry the same call, retry with
modified params, switch method, escalate, or fail fast. The agent only gets
involved for the classes where reflection actually helps.
"""

import time
from enum import Enum


class CorrectionStrategy(str, Enum):
    RETRY_SAME = "retry_same"           # 相同参数重试（瞬态错误）
    RETRY_MODIFIED = "retry_modified"   # 调整参数重试（如超时→加长timeout）
    ALTERNATIVE_METHOD = "alt_method"   # 换工具/换路径（如文件不存在→搜索）
    UPGRADE_MODEL = "upgrade_model"     # 格式/推理错误→换更强模型
    ESCALATE_USER = "escalate_user"     # 权限/业务逻辑不明→用户介入
    FAIL_FAST = "fail_fast"             # 不可恢复错误→立即标记 failed


# Hard cap on the timeout a RETRY_MODIFIED retry may reach. Without it, a
# timeout=30 call retried 3x doubles to 240s and feels hung to the operator.
MAX_RETRY_TIMEOUT = 300


class UserEscalationError(Exception):
    """A human decision is required (permissions, ambiguous business intent)."""


class FatalToolError(Exception):
    """Unrecoverable (e.g. resource exhaustion) — never retry."""


class ErrorClassifier:
    """Deterministic classification — no tokens, no model call."""

    # (exception type | keyword substring, strategy, params)
    RULES = [
        (TimeoutError, CorrectionStrategy.RETRY_MODIFIED, {"timeout_multiplier": 2}),
        (ConnectionResetError, CorrectionStrategy.RETRY_SAME, {}),
        (FileNotFoundError, CorrectionStrategy.ALTERNATIVE_METHOD, {"hint": "use grep to locate"}),
        (PermissionError, CorrectionStrategy.ESCALATE_USER, {}),
        ("invalid json", CorrectionStrategy.UPGRADE_MODEL, {}),
        ("rate limit", CorrectionStrategy.RETRY_MODIFIED, {"backoff": 30}),
        ("syntax error", CorrectionStrategy.UPGRADE_MODEL, {}),
        # MCP Lite's structured timeout error (module D)
        ("mcpservertimeout", CorrectionStrategy.RETRY_MODIFIED, {"backoff": 5, "timeout_multiplier": 2}),
    ]

    @classmethod
    def classify(cls, error: BaseException) -> tuple[CorrectionStrategy, dict]:
        # resource exhaustion from the sandbox (OOM circuit breaker) -> fail fast
        try:
            from ..sandbox import SandboxResourceExhausted
        except ImportError:  # pragma: no cover - sandbox is always importable
            SandboxResourceExhausted = ()
        if isinstance(error, SandboxResourceExhausted):
            return CorrectionStrategy.FAIL_FAST, {}
        if isinstance(error, UserEscalationError):
            return CorrectionStrategy.ESCALATE_USER, {}
        if isinstance(error, FatalToolError):
            return CorrectionStrategy.FAIL_FAST, {}

        # structured errors carry an explicit error_type (e.g. MCPToolError)
        error_type = getattr(error, "error_type", None)
        if error_type:
            lower = str(error_type).lower()
            if "timeout" in lower:
                return CorrectionStrategy.RETRY_MODIFIED, {"backoff": 5, "timeout_multiplier": 2}
            return CorrectionStrategy.RETRY_SAME, {}

        message = str(error).lower()
        for pattern, strategy, params in cls.RULES:
            if isinstance(pattern, type) and isinstance(error, pattern):
                return strategy, params
            if isinstance(pattern, str) and pattern in message:
                return strategy, params
        # 默认：未知错误交给 Agent 反思（更强模型/换方法由 Agent 决定）
        return CorrectionStrategy.UPGRADE_MODEL, {"reason": "unclassified"}


def run_with_correction(
    fn,
    *,
    max_retries: int = 2,
    sleep_fn=time.sleep,
    **kwargs,
):
    """Execute fn(**kwargs), applying deterministic retry strategies.

    * RETRY_SAME / RETRY_MODIFIED: back off and retry (RETRY_MODIFIED also
      extends a `timeout` kwarg when the call has one);
    * ESCALATE_USER / FAIL_FAST: raise UserEscalationError / FatalToolError;
    * everything else surfaces so the agent can reflect and choose an
      alternative method.

    The agent's `_exec_tool` routes every tool call through this wrapper.
    """
    attempt = 0
    while True:
        try:
            return fn(**kwargs)
        except Exception as exc:
            strategy, params = ErrorClassifier.classify(exc)
            attempt += 1

            if strategy in (
                CorrectionStrategy.RETRY_SAME,
                CorrectionStrategy.RETRY_MODIFIED,
            ) and attempt <= max_retries:
                if (
                    strategy == CorrectionStrategy.RETRY_MODIFIED
                    and "timeout" in kwargs
                ):
                    kwargs = dict(kwargs)
                    # cap the exponential growth so retries never feel hung
                    new_timeout = int(
                        kwargs["timeout"] * params.get("timeout_multiplier", 2)
                    )
                    kwargs["timeout"] = min(max(new_timeout, 1), MAX_RETRY_TIMEOUT)
                sleep_fn(params.get("backoff", 2 ** (attempt - 1)))
                continue

            if strategy == CorrectionStrategy.ESCALATE_USER:
                raise UserEscalationError(f"需要用户授权或澄清: {exc}") from exc
            if strategy == CorrectionStrategy.FAIL_FAST:
                raise FatalToolError(f"不可恢复错误: {exc}") from exc

            raise  # surface: agent reflects (alt_method / upgrade_model / retries exhausted)
