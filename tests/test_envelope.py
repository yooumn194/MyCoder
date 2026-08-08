"""Tests for the Pydantic Subagent Result Envelope (RFC v1.0.1 canonical)."""

import pytest
from pydantic import ValidationError

from corecoder.contracts import (
    GeneralResult,
    parse_envelope,
)

INSTANCE = "11111111-2222-3333-4444-555555555555"


def _meta(**kw) -> dict:
    base = {
        "task_id": "task-1",
        "subagent_name": "explorer",
        "subagent_instance_id": INSTANCE,
        "started_at": "2026-08-08T00:00:00Z",
        "finished_at": "2026-08-08T00:00:01Z",
        "duration_ms": 1000,
    }
    base.update(kw)
    return base


def _error(**kw) -> dict:
    err = {"code": "X", "category": "permanent", "retryable": False, "message": "boom"}
    err.update(kw)
    return err


def _envelope(**kw) -> dict:
    base = {
        "meta": _meta(),
        "status": "success",
        "summary": "done",
        "confidence": "high",
        "result": {"type": "general", "output": "ok"},
    }
    base.update(kw)
    return base


def test_valid_success_parses():
    env = parse_envelope(_envelope())
    assert env.status == "success"
    assert env.result.type == "general"


def test_success_with_error_rejected():
    with pytest.raises(ValidationError, match="success.*error"):
        parse_envelope(_envelope(error=_error()))


def test_partial_requires_all_fields():
    valid = _envelope(status="partial", confidence="medium",
                      completeness_ratio=0.5, error=_error())
    assert parse_envelope(valid).status == "partial"


@pytest.mark.parametrize("bad", [0.0, 1.0])
def test_partial_ratio_strict_bounds(bad):
    with pytest.raises(ValidationError, match="0.0 < x < 1.0"):
        parse_envelope(_envelope(status="partial", confidence="medium",
                                 completeness_ratio=bad, error=_error()))


def test_partial_missing_ratio_rejected():
    with pytest.raises(ValidationError, match="completeness_ratio"):
        parse_envelope(_envelope(status="partial", confidence="medium", error=_error()))


def test_failed_requires_error_no_result():
    ok = _envelope(status="failed", confidence="low", result=None, error=_error())
    assert parse_envelope(ok).status == "failed"
    with pytest.raises(ValidationError):
        parse_envelope(_envelope(status="failed", confidence="low", result=None))


def test_cancelled_error_optional_and_code_checked():
    ok = _envelope(status="cancelled", confidence="low", result=None)
    assert parse_envelope(ok).status == "cancelled"
    with_code = _envelope(status="cancelled", confidence="low", result=None,
                          error=_error(code="TASK_CANCELLED", category="system_constraint", retryable=True))
    assert parse_envelope(with_code).status == "cancelled"
    with pytest.raises(ValidationError, match="TASK_CANCELLED"):
        parse_envelope(_envelope(status="cancelled", confidence="low", result=None, error=_error()))


def test_artifacts_max_100_enforced():
    data = _envelope(artifacts=[{"path": f"f{i}.py", "action": "created"} for i in range(101)])
    with pytest.raises(ValidationError):
        parse_envelope(data)


def test_summary_max_500_enforced():
    with pytest.raises(ValidationError):
        parse_envelope(_envelope(summary="x" * 501))


def test_instance_id_required():
    meta = _meta()
    meta.pop("subagent_instance_id")
    with pytest.raises(ValidationError, match="subagent_instance_id"):
        parse_envelope(_envelope(meta=meta))


def test_suggestion_params_string_array():
    env = _envelope(suggested_next_step={"type": "continue", "params": {"files": ["a.py"]}})
    assert parse_envelope(env).suggested_next_step.params["files"] == ["a.py"]


def test_result_type_union_discriminates():
    env = parse_envelope(_envelope(result={"type": "general", "output": "hello"}))
    assert isinstance(env.result, GeneralResult)
