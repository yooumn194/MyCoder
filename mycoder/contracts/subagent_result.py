"""Subagent Result Contract v1.0.1 — frozen.

See RFC: Subagent Result Contract Schema v1.0 (+ the v1.0.1 implementation
patch). This module is the canonical Python validator. It HARD-FAILS on
violations — no silent normalization. Rewriting a subagent's status or
completeness_ratio would mask a prompt defect upstream, so the contract layer
keeps data pure and returns 400-style errors instead.

Frozen decision (ADR-001): no SDK-layer JSON auto-repair. Validation failure
is a real error the orchestrator decides how to handle.
"""

import json
import re
import uuid
from typing import Any

# ---------------------------------------------------------------------------
# Frozen constants (v1.0.1)
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "1.0.1"

VALID_STATUSES = {"success", "partial", "failed", "cancelled"}
VALID_CONFIDENCE = {"high", "medium", "low"}
VALID_CATEGORIES = {"transient", "permanent", "user_input_required", "system_constraint"}
VALID_RESULT_TYPES = {
    "code_analysis",
    "code_generation",
    "code_exploration",
    "plan",
    "review",
    "general",
}
VALID_ARTIFACT_ACTIONS = {"created", "modified", "deleted"}
VALID_SUGGESTION_TYPES = {"retry", "continue", "escalate", "stop"}

MAX_SUMMARY_CHARS = 500
MAX_ARTIFACTS = 100          # v1.0.1: hard cap (was unbounded in v1.0)
MAX_PARAMS_ARRAY_ITEMS = 20  # v1.0.1: string arrays only, max 20

UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
_ABSOLUTE_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")

# error.category -> retryable consistency (ADR-003, combination legality).
CATEGORY_RETRYABLE: dict[str, bool] = {
    "transient": True,             # LLM rate limit, network blip, temp lock
    "permanent": False,            # schema violation, permission denied, missing file
    "user_input_required": False,  # retrying before a human decides is pointless
    "system_constraint": True,     # only after resources free up
}


class ContractValidationError(Exception):
    """Raised by validate_or_raise when the result violates the contract."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("; ".join(errors))


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class SubagentResultValidator:
    """Enforces the Subagent Result Contract v1.0.1."""

    def __init__(self, *, allow_missing_instance_id: bool = False) -> None:
        # allow_missing_instance_id exists for tests / transitional consumers;
        # production (the orchestrator) always requires it.
        self._allow_missing_instance_id = allow_missing_instance_id

    def validate(self, result: dict) -> list[str]:
        errors: list[str] = []
        self._validate_meta(result.get("meta"), errors)
        self._validate_summary(result, errors)
        self._validate_status_matrix(result, errors)
        self._validate_error(result, errors)
        self._validate_result_payload(result, errors)
        self._validate_artifacts(result, errors)
        self._validate_suggestion(result, errors)
        return errors

    def validate_or_raise(self, result: dict) -> None:
        errors = self.validate(result)
        if errors:
            raise ContractValidationError(errors)

    # ------------------------------------------------------------- internals

    def _validate_meta(self, meta: Any, errors: list[str]) -> None:
        if not isinstance(meta, dict):
            errors.append("meta must be an object")
            return
        if not meta.get("task_id"):
            errors.append("meta.task_id is required (globally unique)")
        if not meta.get("subagent_name"):
            errors.append("meta.subagent_name is required")
        instance_id = meta.get("subagent_instance_id")
        if not self._allow_missing_instance_id and not instance_id:
            errors.append("meta.subagent_instance_id is required (UUID v4 injected by orchestrator)")
        elif instance_id and not UUID_RE.match(str(instance_id)):
            errors.append(f"meta.subagent_instance_id must be a UUID v4, got {instance_id!r}")
        for field in ("started_at", "finished_at"):
            if not meta.get(field):
                errors.append(f"meta.{field} is required (ISO 8601)")
        if not isinstance(meta.get("duration_ms"), (int, float)) or isinstance(meta.get("duration_ms"), bool):
            errors.append("meta.duration_ms must be a number")

    def _validate_summary(self, result: dict, errors: list[str]) -> None:
        summary = result.get("summary")
        if not isinstance(summary, str) or not summary:
            errors.append("summary must be a non-empty string")
        elif len(summary) > MAX_SUMMARY_CHARS:
            errors.append(f"summary exceeds {MAX_SUMMARY_CHARS} chars ({len(summary)})")

    def _validate_status_matrix(self, result: dict, errors: list[str]) -> None:
        """The ONLY legal status/confidence/result/error/ratio combinations."""
        status = result.get("status")
        if status not in VALID_STATUSES:
            errors.append(f"status must be one of {sorted(VALID_STATUSES)}, got {status!r}")
            return
        # a null/absent result or error is "not carrying it" — failed/cancelled
        # legitimately omit these fields entirely.
        has_result = result.get("result") is not None
        has_error = result.get("error") is not None
        ratio = result.get("completeness_ratio")
        conf = result.get("confidence")

        if status in ("success", "partial") and not has_result:
            errors.append(f"status={status} requires result")
        if status == "success":
            if has_error:
                errors.append("status=success must NOT carry error (even an empty one)")
            if conf not in ("high", "medium"):
                errors.append("status=success confidence must be high or medium")
            if ratio is not None:
                errors.append("status=success must not carry completeness_ratio")
        elif status == "partial":
            if not has_error:
                errors.append("status=partial requires error (reason for the missing part)")
            if conf not in ("medium", "low"):
                errors.append("status=partial confidence must be medium or low")
            # v1.0.1 verdict: strict bounds — 0.0/1.0 are rejected, not normalized.
            if not isinstance(ratio, (int, float)) or isinstance(ratio, bool):
                errors.append("status=partial requires completeness_ratio")
            elif not (0.0 < ratio < 1.0):
                errors.append(f"completeness_ratio must satisfy 0.0 < x < 1.0 for partial, got {ratio!r}")
        elif status == "failed":
            if has_result:
                errors.append("status=failed must NOT carry result")
            if not has_error:
                errors.append("status=failed requires error")
            if conf != "low":
                errors.append("status=failed confidence must be low")
            if ratio is not None:
                errors.append("status=failed must not carry completeness_ratio")
        elif status == "cancelled":
            if has_result:
                errors.append("status=cancelled must NOT carry result")
            # v1.0.1: error is OPTIONAL for cancelled (a user cancel is not an error);
            # when present its code must be TASK_CANCELLED.
            if has_error and result["error"].get("code") != "TASK_CANCELLED":
                errors.append("cancelled error.code must be 'TASK_CANCELLED'")
            if conf != "low":
                errors.append("status=cancelled confidence must be low")
            if ratio is not None:
                errors.append("status=cancelled must not carry completeness_ratio")

    def _validate_error(self, result: dict, errors: list[str]) -> None:
        err = result.get("error")
        if err is None:
            return
        if not isinstance(err, dict):
            errors.append("error must be an object")
            return
        category = err.get("category")
        retryable = err.get("retryable")
        if category not in VALID_CATEGORIES:
            errors.append(f"error.category must be one of {sorted(VALID_CATEGORIES)}, got {category!r}")
        if not isinstance(retryable, bool):
            errors.append("error.retryable must be a boolean")
        else:
            expected = CATEGORY_RETRYABLE.get(category)
            if expected is not None and retryable != expected:
                errors.append(
                    f"error.retryable={retryable} conflicts with category={category} "
                    f"(expected {expected})"
                )
        if not isinstance(err.get("code"), str) or not err["code"]:
            errors.append("error.code must be a non-empty string")
        if not isinstance(err.get("message"), str) or not err["message"]:
            errors.append("error.message must be a non-empty string")

    def _validate_result_payload(self, result: dict, errors: list[str]) -> None:
        payload = result.get("result")
        if payload is None:
            return
        if not isinstance(payload, dict):
            errors.append("result must be an object")
            return
        rtype = payload.get("type")
        if rtype not in VALID_RESULT_TYPES:
            errors.append(f"result.type must be one of {sorted(VALID_RESULT_TYPES)}, got {rtype!r}")

    def _validate_artifacts(self, result: dict, errors: list[str]) -> None:
        artifacts = result.get("artifacts")
        if artifacts is None:
            return
        if not isinstance(artifacts, list):
            errors.append("artifacts must be an array")
            return
        if len(artifacts) > MAX_ARTIFACTS:
            errors.append(f"artifacts exceeds maxItems={MAX_ARTIFACTS} ({len(artifacts)} items)")
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                errors.append("each artifact must be an object")
                continue
            path = artifact.get("path")
            if not isinstance(path, str) or not path:
                errors.append("artifact.path must be a non-empty string")
                continue
            # workspace-relative only: no absolute, no ../ traversal
            if path.startswith("/") or _ABSOLUTE_PATH_RE.match(path):
                errors.append(f"artifact.path must be workspace-relative, got absolute: {path!r}")
            if ".." in path.split("/"):
                errors.append(f"artifact.path must not traverse ('..'): {path!r}")
            if artifact.get("action") not in VALID_ARTIFACT_ACTIONS:
                errors.append(
                    f"artifact.action must be one of {sorted(VALID_ARTIFACT_ACTIONS)}, "
                    f"got {artifact.get('action')!r}"
                )

    def _validate_suggestion(self, result: dict, errors: list[str]) -> None:
        sug = result.get("suggested_next_step")
        if sug is None:
            return
        if not isinstance(sug, dict):
            errors.append("suggested_next_step must be an object")
            return
        if sug.get("type") not in VALID_SUGGESTION_TYPES:
            errors.append(
                f"suggested_next_step.type must be one of {sorted(VALID_SUGGESTION_TYPES)}, "
                f"got {sug.get('type')!r}"
            )
        params = sug.get("params")
        if params is None:
            return
        if not isinstance(params, dict):
            errors.append("suggested_next_step.params must be an object")
            return
        for key, value in params.items():
            if isinstance(value, list):
                if len(value) > MAX_PARAMS_ARRAY_ITEMS:
                    errors.append(f"params.{key} array exceeds maxItems={MAX_PARAMS_ARRAY_ITEMS}")
                if not all(isinstance(x, str) for x in value):
                    errors.append(f"params.{key} array must contain only strings")
            elif not isinstance(value, (str, int, float, bool)):
                errors.append(f"params.{key} must be a scalar or a string array")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def new_instance_id() -> str:
    """A fresh UUID v4 for meta.subagent_instance_id (orchestrator injects this)."""
    return str(uuid.uuid4())


# RFC §行为契约: error.category is an INSTRUCTION to the orchestrator, mapped
# onto the Phase 3 self-correction strategies.
_CATEGORY_STRATEGY = {
    "transient": "retry_same",              # SHOULD exponential-backoff retry (max 3)
    "permanent": "fail_fast",               # MUST terminate, never auto-retry
    "user_input_required": "escalate_user",  # MUST pause for human-in-the-loop
    "system_constraint": "retry_modified",   # SHOULD back off, then retry
}


def category_to_strategy(category: str) -> str:
    """error.category -> CorrectionStrategy name (deterministic routing)."""
    return _CATEGORY_STRATEGY.get(category, "upgrade_model")


def parse_result(text: str) -> dict:
    """Extract the JSON object from a subagent's response text (lenient).

    Finds the first '{' .. last '}' so surrounding prose is tolerated — but if
    the JSON itself is malformed we let json.JSONDecodeError propagate (ADR-001:
    no auto-repair).
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise json.JSONDecodeError("no JSON object found", text, 0)
    return json.loads(text[start : end + 1])


def migrate_v0_1_to_v1_0(v0_1_result: dict) -> dict:
    """Transitional adapter: v0.1 consumers' results -> v1.0 structure.

    Removes every internal text-summary field (ADR-002: redundant summaries
    were the top source of multi-agent inconsistency), guarantees a top-level
    summary <= 500 chars, and stamps the schema version.
    """
    v1 = dict(v0_1_result)
    inner = v1.get("result")
    if isinstance(inner, dict):
        for key in (
            "summary", "detail_summary", "planning_notes",
            "review_summary", "exploration_summary", "generated_summary",
        ):
            inner.pop(key, None)
    if "summary" not in v1:
        v1["summary"] = str(v0_1_result.get("summary", "No summary provided"))
    if len(v1["summary"]) > MAX_SUMMARY_CHARS:
        v1["summary"] = v1["summary"][: MAX_SUMMARY_CHARS - 3] + "..."
    v1["schema_version"] = "1.0"
    return v1
