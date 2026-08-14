"""RFC v1.0.1 Subagent Result Envelope — Pydantic canonical contract.

This is the Phase 4 canonical validator used by the multi-agent Runner and
Orchestrator. It enforces the frozen state-combination matrix at model level
(Pydantic model_validator), so no invalid envelope can ever be constructed or
accepted.

Relationship to mycoder/contracts/subagent_result.py: that module is the
lighter manual validator used by the legacy AgentTool text path; this Pydantic
envelope is the source of truth for the Phase 4 agent system.
"""

import uuid
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

__all__ = [
    "Artifact",
    "ErrorObject",
    "ExplorationResult",
    "GeneralResult",
    "ImplementationResult",
    "Meta",
    "PlanResult",
    "ResultPayload",
    "ReviewResult",
    "SubagentResultEnvelope",
    "Suggestion",
    "ValidationError",
]

from pydantic import ValidationError  # noqa: E402  (re-exported for callers)

# ===== 元数据（v1.0.1：含 subagent_instance_id） =====


class Meta(BaseModel):
    task_id: str
    subagent_name: str
    subagent_instance_id: str = Field(..., description="UUID v4，由 Orchestrator 注入，用于幂等去重")
    session_id: Optional[str] = None
    trace_context: Optional[dict[str, Any]] = None
    started_at: str  # ISO 8601
    finished_at: str  # ISO 8601
    duration_ms: int
    parent_tool_use_id: Optional[str] = None
    context_used: Optional[list[str]] = None


# ===== 错误对象 =====


class ErrorObject(BaseModel):
    code: str
    category: Literal["transient", "permanent", "user_input_required", "system_constraint"]
    retryable: bool
    message: str
    details: Optional[Any] = None


# ===== 制品（v1.0.1：maxItems=100） =====


class Artifact(BaseModel):
    path: str = Field(..., description="workspace-relative，禁止 ../ 和绝对路径")
    action: Literal["created", "modified", "deleted"]
    hash: Optional[str] = None


# ===== 下一步建议（v1.0.1：params 放宽为含一维字符串数组） =====


class Suggestion(BaseModel):
    type: Literal["retry", "continue", "escalate", "stop"]
    target: Optional[str] = None
    params: Optional[dict[str, Union[str, int, float, bool, list[str]]]] = Field(
        None,
        description="仅含标量或一维字符串数组（max 20 项）",
    )
    reason: Optional[str] = None


# ===== 各 Subagent 专用的 ResultPayload =====


class ExplorationResult(BaseModel):
    type: Literal["code_exploration"]
    files_found: list[dict[str, Any]]
    patterns_searched: list[str]
    total_matches: int


class PlanResult(BaseModel):
    type: Literal["plan"]
    plan: dict[str, Any]


class ImplementationResult(BaseModel):
    type: Literal["code_generation"]
    files: list[dict[str, Any]]
    test_results: Optional[list[dict[str, Any]]] = None


class ReviewResult(BaseModel):
    type: Literal["review"]
    verdict: Literal["pass", "fail", "conditional"]
    checks: list[dict[str, Any]]
    recommendations: list[str]


class GeneralResult(BaseModel):
    type: Literal["general"]
    output: str
    structured_output: Optional[dict[str, Any]] = None


ResultPayload = Union[
    ExplorationResult,
    PlanResult,
    ImplementationResult,
    ReviewResult,
    GeneralResult,
]


class SubagentResultEnvelope(BaseModel):
    """RFC v1.0.1 标准信封 — canonical, validated at construction."""

    schema_version: Literal["1.0.1"] = "1.0.1"
    meta: Meta
    status: Literal["success", "partial", "failed", "cancelled"]
    summary: str = Field(..., max_length=500)
    confidence: Literal["high", "medium", "low"]
    result: Optional[ResultPayload] = None
    error: Optional[ErrorObject] = None
    completeness_ratio: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="仅 status=partial 时必填，必须满足 0.0 < x < 1.0",
    )
    artifacts: Optional[list[Artifact]] = Field(None, max_length=100)
    usage: Optional[dict[str, int]] = None
    suggested_next_step: Optional[Suggestion] = None

    @model_validator(mode="after")
    def validate_status_combinations(self) -> "SubagentResultEnvelope":
        if self.status == "partial":
            if self.completeness_ratio is None:
                raise ValueError("status=partial 时 completeness_ratio 必填")
            if not (0.0 < self.completeness_ratio < 1.0):
                raise ValueError("status=partial 时 completeness_ratio 必须满足 0.0 < x < 1.0")
            if self.result is None:
                raise ValueError("status=partial 时 result 必填")
            if self.error is None:
                raise ValueError("status=partial 时 error 必填")
        elif self.status == "success":
            if self.result is None:
                raise ValueError("status=success 时 result 必填")
            if self.error is not None:
                raise ValueError("status=success 时 error 必须为空")
            if self.completeness_ratio is not None:
                raise ValueError("status=success 时 completeness_ratio 必须为空")
        elif self.status == "failed":
            if self.error is None:
                raise ValueError("status=failed 时 error 必填")
            if self.result is not None:
                raise ValueError("status=failed 时 result 必须为空")
            if self.completeness_ratio is not None:
                raise ValueError("status=failed 时 completeness_ratio 必须为空")
        elif self.status == "cancelled":
            if self.result is not None:
                raise ValueError("status=cancelled 时 result 必须为空")
            if self.error is not None and self.error.code != "TASK_CANCELLED":
                raise ValueError("status=cancelled 且 error 存在时，code 必须为 'TASK_CANCELLED'")
            if self.completeness_ratio is not None:
                raise ValueError("status=cancelled 时 completeness_ratio 必须为空")
        return self


def new_envelope_instance_id() -> str:
    """A fresh UUID v4 for meta.subagent_instance_id."""
    return str(uuid.uuid4())


def parse_envelope(data: dict) -> SubagentResultEnvelope:
    """Validate an envelope from a dict, letting pydantic.ValidationError propagate."""
    return SubagentResultEnvelope.model_validate(data)
