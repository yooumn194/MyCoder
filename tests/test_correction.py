"""Tests for the Phase 3 self-correction loop (deterministic error -> strategy)."""

import pytest

from corecoder.tools.correction import (
    CorrectionStrategy,
    ErrorClassifier,
    FatalToolError,
    UserEscalationError,
    run_with_correction,
)


# ---------------------------------------------------------------------------
# ErrorClassifier: deterministic mapping
# ---------------------------------------------------------------------------

def test_classify_timeout_retry_modified():
    strategy, params = ErrorClassifier.classify(TimeoutError("took too long"))
    assert strategy == CorrectionStrategy.RETRY_MODIFIED
    assert params["timeout_multiplier"] == 2


def test_classify_connection_reset_retry_same():
    strategy, _ = ErrorClassifier.classify(ConnectionResetError("reset"))
    assert strategy == CorrectionStrategy.RETRY_SAME


def test_classify_file_not_found_alt_method():
    strategy, params = ErrorClassifier.classify(FileNotFoundError("x.txt"))
    assert strategy == CorrectionStrategy.ALTERNATIVE_METHOD
    assert "grep" in params["hint"]


def test_classify_permission_escalate_user():
    strategy, _ = ErrorClassifier.classify(PermissionError("denied"))
    assert strategy == CorrectionStrategy.ESCALATE_USER


def test_classify_keyword_rules():
    assert ErrorClassifier.classify(ValueError("invalid JSON at line 1"))[0] == CorrectionStrategy.UPGRADE_MODEL
    assert ErrorClassifier.classify(RuntimeError("rate limit exceeded"))[0] == CorrectionStrategy.RETRY_MODIFIED
    assert ErrorClassifier.classify(SyntaxError("syntax error near x"))[0] == CorrectionStrategy.UPGRADE_MODEL


def test_classify_unknown_defaults_to_upgrade():
    strategy, params = ErrorClassifier.classify(ArithmeticError("weird"))
    assert strategy == CorrectionStrategy.UPGRADE_MODEL
    assert params["reason"] == "unclassified"


def test_classify_sandbox_oom_fail_fast():
    from corecoder.sandbox import SandboxResourceExhausted

    strategy, _ = ErrorClassifier.classify(
        SandboxResourceExhausted("container was OOM-killed 3 times")
    )
    assert strategy == CorrectionStrategy.FAIL_FAST


def test_classify_mcp_error_types():
    """Every MCP error_type maps to the correct correction strategy."""
    from corecoder.mcp.errors import MCPToolError

    cases = {
        "MCPInvalidRequest": CorrectionStrategy.ESCALATE_USER,
        "MCPInvalidParams": CorrectionStrategy.ESCALATE_USER,
        "MCPMethodNotFound": CorrectionStrategy.FAIL_FAST,
        "MCPInternalError": CorrectionStrategy.RETRY_SAME,
        "MCPServerError": CorrectionStrategy.ALTERNATIVE_METHOD,
        "MCPHTTPError": CorrectionStrategy.RETRY_MODIFIED,
        "MCPUnknownError": CorrectionStrategy.UPGRADE_MODEL,
        "MCPServerTimeout": CorrectionStrategy.RETRY_MODIFIED,
    }
    for error_type, expected in cases.items():
        err = MCPToolError(error_type, "fs", "tool", "boom")
        assert ErrorClassifier.classify(err)[0] == expected, error_type


# ---------------------------------------------------------------------------
# run_with_correction: the retry loop
# ---------------------------------------------------------------------------

def test_retry_same_eventually_succeeds():
    calls = []

    def _flaky(x):
        calls.append(x)
        if len(calls) < 3:
            raise ConnectionResetError("transient")
        return f"ok:{x}"

    result = run_with_correction(_flaky, x=1, sleep_fn=lambda _: None)
    assert result == "ok:1"
    assert len(calls) == 3  # two retries


def test_retry_modified_extends_timeout():
    seen = []

    def _tool(**kwargs):
        seen.append(kwargs.get("timeout"))
        if len(seen) == 1:
            raise TimeoutError("too slow")
        return "done"

    run_with_correction(_tool, timeout=30, sleep_fn=lambda _: None)
    assert seen[0] == 30
    assert seen[1] == 60  # timeout_multiplier 2


def test_retry_modified_timeout_is_capped():
    """Exponential timeout growth must never exceed MAX_RETRY_TIMEOUT."""
    from corecoder.tools.correction import MAX_RETRY_TIMEOUT

    seen = []

    def _tool(**kwargs):
        seen.append(kwargs.get("timeout"))
        if len(seen) < 3:
            raise TimeoutError("slow")
        return "done"

    run_with_correction(_tool, timeout=200, sleep_fn=lambda _: None)
    assert seen == [200, MAX_RETRY_TIMEOUT, MAX_RETRY_TIMEOUT]  # 200->400->capped


def test_gives_up_after_max_retries():
    calls = []

    def _always_fails():
        calls.append(1)
        raise ConnectionResetError("persistent")

    with pytest.raises(ConnectionResetError):
        run_with_correction(_always_fails, max_retries=2, sleep_fn=lambda _: None)
    assert len(calls) == 3  # 1 attempt + 2 retries


def test_permission_error_escalates_to_user():
    def _denied():
        raise PermissionError("no write access")

    with pytest.raises(UserEscalationError) as exc:
        run_with_correction(_denied, sleep_fn=lambda _: None)
    assert "授权或澄清" in str(exc.value)


def test_oom_fails_fast():
    from corecoder.sandbox import SandboxResourceExhausted

    def _oom():
        raise SandboxResourceExhausted("OOM")

    with pytest.raises(FatalToolError) as exc:
        run_with_correction(_oom, sleep_fn=lambda _: None)
    assert "不可恢复" in str(exc.value)


def test_alt_method_surfaces_to_agent():
    """FileNotFoundError must NOT be retried — it surfaces for the agent to reflect."""

    def _missing():
        raise FileNotFoundError("no such file")

    with pytest.raises(FileNotFoundError):
        run_with_correction(_missing, sleep_fn=lambda _: None)
