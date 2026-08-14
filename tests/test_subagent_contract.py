"""Tests for the Subagent Result Contract validator + orchestrator integration."""

import json

import pytest

from mycoder.contracts import (
    SubagentResultValidator,
    category_to_strategy,
    migrate_v0_1_to_v1_0,
    parse_result,
)
from mycoder.tools.agent import AgentTool

INSTANCE = "11111111-2222-3333-4444-555555555555"


@pytest.fixture(autouse=True)
def _clear_result_cache():
    """AgentTool's idempotency cache is class-level; isolate it per test."""
    AgentTool._result_cache.clear()
    yield
    AgentTool._result_cache.clear()


def _result(**overrides) -> dict:
    base = {
        "meta": {
            "task_id": "task-1",
            "subagent_name": "explorer",
            "subagent_instance_id": INSTANCE,
            "started_at": "2026-08-08T00:00:00Z",
            "finished_at": "2026-08-08T00:00:01Z",
            "duration_ms": 1000,
        },
        "status": "success",
        "summary": "found the bug",
        "confidence": "high",
        "result": {"type": "code_analysis", "findings": [], "files_analyzed": ["a.py"]},
    }
    base.update(overrides)
    return base


def _err(**kw) -> dict:
    err = {"code": "FILE_NOT_FOUND", "category": "permanent", "retryable": False, "message": "gone"}
    err.update(kw)
    return err


# ---------------------------------------------------------------------------
# state matrix
# ---------------------------------------------------------------------------

def test_valid_success_passes():
    assert SubagentResultValidator().validate(_result()) == []


def test_success_with_error_rejected():
    errors = SubagentResultValidator().validate(_result(error=_err()))
    assert any("must NOT carry error" in e for e in errors)


def test_partial_requires_error_and_ratio():
    r = _result(status="partial", confidence="medium", completeness_ratio=0.5, error=_err())
    assert SubagentResultValidator().validate(r) == []


def test_partial_missing_ratio_rejected():
    r = _result(status="partial", confidence="medium", error=_err())
    errors = SubagentResultValidator().validate(r)
    assert any("completeness_ratio" in e for e in errors)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_partial_ratio_bounds_rejected(bad):
    """0.0/1.0 are rejected, not normalized (final-verdict decision #1)."""
    r = _result(status="partial", confidence="medium", completeness_ratio=bad, error=_err())
    errors = SubagentResultValidator().validate(r)
    assert any("0.0 < x < 1.0" in e for e in errors)


def test_failed_requires_error_and_no_result():
    r = _result(status="failed", confidence="low", result=None, error=_err())
    assert SubagentResultValidator().validate(r) == []
    r2 = _result(status="failed", confidence="low", result=None)  # no error
    assert any("requires error" in e for e in SubagentResultValidator().validate(r2))


def test_cancelled_error_optional():
    r = _result(status="cancelled", confidence="low", result=None)
    assert SubagentResultValidator().validate(r) == []


def test_cancelled_with_error_code():
    r = _result(status="cancelled", confidence="low", result=None,
                error=_err(code="TASK_CANCELLED", category="system_constraint", retryable=True))
    assert SubagentResultValidator().validate(r) == []
    bad = _result(status="cancelled", confidence="low", result=None, error=_err())
    assert any("TASK_CANCELLED" in e for e in SubagentResultValidator().validate(bad))


def test_success_requires_result():
    errors = SubagentResultValidator().validate(_result(result=None))
    assert any("requires result" in e for e in errors)


# ---------------------------------------------------------------------------
# error.category / retryable consistency + behavior contract
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("category,retryable", [
    ("transient", True), ("permanent", False),
    ("user_input_required", False), ("system_constraint", True),
])
def test_category_retryable_consistency_ok(category, retryable):
    r = _result(status="failed", confidence="low", result=None,
                error=_err(category=category, retryable=retryable))
    assert SubagentResultValidator().validate(r) == []


def test_category_retryable_conflict_rejected():
    r = _result(status="failed", confidence="low", result=None,
                error=_err(category="permanent", retryable=True))  # permanent must be False
    errors = SubagentResultValidator().validate(r)
    assert any("conflicts with category" in e for e in errors)


def test_category_to_strategy_routes_to_correction():
    assert category_to_strategy("transient") == "retry_same"
    assert category_to_strategy("permanent") == "fail_fast"
    assert category_to_strategy("user_input_required") == "escalate_user"
    assert category_to_strategy("system_constraint") == "retry_modified"
    assert category_to_strategy("unknown") == "upgrade_model"


# ---------------------------------------------------------------------------
# artifacts: maxItems + workspace-relative paths
# ---------------------------------------------------------------------------

def test_artifacts_over_max_rejected():
    r = _result(artifacts=[{"path": f"f{i}.py", "action": "created"} for i in range(101)])
    errors = SubagentResultValidator().validate(r)
    assert any("maxItems=100" in e for e in errors)


@pytest.mark.parametrize("bad_path", ["/etc/passwd", "C:\\evil.txt", "../outside.py"])
def test_artifacts_absolute_or_traversal_rejected(bad_path):
    r = _result(artifacts=[{"path": bad_path, "action": "created"}])
    errors = SubagentResultValidator().validate(r)
    assert any("workspace-relative" in e or "traverse" in e for e in errors)


def test_artifacts_valid_path_ok():
    r = _result(artifacts=[{"path": "src/auth/handler.py", "action": "modified"}])
    assert SubagentResultValidator().validate(r) == []


# ---------------------------------------------------------------------------
# summary / meta / suggestion
# ---------------------------------------------------------------------------

def test_summary_over_500_rejected():
    r = _result(summary="x" * 501)
    assert any("500" in e for e in SubagentResultValidator().validate(r))


def test_instance_id_required_by_default():
    r = _result()
    r["meta"] = {k: v for k, v in r["meta"].items() if k != "subagent_instance_id"}
    assert any("subagent_instance_id" in e for e in SubagentResultValidator().validate(r))


def test_instance_id_bad_uuid_rejected():
    r = _result()
    r["meta"]["subagent_instance_id"] = "not-a-uuid"
    assert any("UUID" in e for e in SubagentResultValidator().validate(r))


def test_params_string_array_allowed():
    r = _result(suggested_next_step={"type": "continue", "params": {"files": ["a.py", "b.py"]}})
    assert SubagentResultValidator().validate(r) == []


def test_params_nested_array_rejected():
    r = _result(suggested_next_step={"type": "continue", "params": {"matrix": [["a"]]}})
    errors = SubagentResultValidator().validate(r)
    assert any("params.matrix" in e for e in errors)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def test_parse_result_extracts_json_from_prose():
    text = "Here is my report:\n" + json.dumps(_result()) + "\nHope that helps."
    assert parse_result(text)["status"] == "success"


def test_migrate_v0_1_to_v1_0_strips_internal_summaries():
    v0 = {
        "summary": "long summary here",
        "result": {"type": "code_analysis", "summary": "internal", "findings": []},
    }
    v1 = migrate_v0_1_to_v1_0(v0)
    assert "summary" not in v1["result"]  # internal summary stripped
    assert v1["schema_version"] == "1.0"
    assert v1["summary"] == "long summary here"


# ---------------------------------------------------------------------------
# orchestrator integration (AgentTool._finalize)
# ---------------------------------------------------------------------------

def _tool_finalize(raw, task_id="t-1"):
    return AgentTool()._finalize(task_id, INSTANCE, raw, 100.0, 0.0)


def test_finalize_valid_contract():
    out = _tool_finalize(json.dumps(_result()))
    assert "Sub-agent success" in out
    assert "artifacts" not in out  # no artifacts in this result


def test_finalize_contract_violation_is_permanent():
    out = _tool_finalize(json.dumps(_result(error=_err())))  # success + error
    assert "contract violation (permanent)" in out
    assert "must NOT carry error" in out


def test_finalize_plain_text_backward_compatible():
    out = _tool_finalize("just did the research\nmore text")
    assert "Sub-agent completed" in out


def test_finalize_idempotent_dedup():
    tool = AgentTool()
    raw = json.dumps(_result())
    first = tool._finalize("dup-1", INSTANCE, raw, 100.0, 0.0)
    second = tool._finalize("dup-1", INSTANCE, "unused different raw", 100.0, 0.0)
    assert "cached" in second
    assert first != second  # the cache hit doesn't re-parse
