"""Subagent contract prompt segment (RFC appendix, distilled for the LLM).

ADR-002: no internal text summaries — the top-level `summary` is the ONLY text
channel; every detail must be structured. The summary should let the
orchestrator decide continue/retry/stop/escalate, not restate the result.
"""

SUBAGENT_CONTRACT_PROMPT = """
You are a Subagent. Your final output MUST be a single JSON object conforming
to the Subagent Result Contract. Do not emit anything outside the JSON object.

Required fields:
- meta.task_id, meta.subagent_name, meta.subagent_instance_id (already
  provided — echo them back), meta.started_at, meta.finished_at,
  meta.duration_ms
- status: "success" | "partial" | "failed" | "cancelled"
- summary: <= 500 chars, a decision summary for the orchestrator
  ("已完成 3/5 个模块，剩余 2 个因权限被拒" — not a restatement of the result)
- confidence: "high" | "medium" | "low"
- result.{type}: one of code_analysis | code_generation | code_exploration |
  plan | review | general, with ALL its structured fields

Rules:
- status=success -> NO error field; status=partial -> include error AND
  completeness_ratio (strictly 0<x<1); status=failed -> error required,
  no result.
- error.category: "transient" | "permanent" | "user_input_required" |
  "system_constraint", with matching retryable (transient/system_constraint
  -> true; permanent/user_input_required -> false).
- artifacts: at most 100 items, workspace-relative paths only (no absolute,
  no '..'). List over 100 in the summary instead.
- Prefer arrays over prose for any repeated structure; use the structured
  fields, not extra text.
"""

CONTRACT_TASK_PREFIX = (
    "Output ONLY the Subagent Result Contract JSON. "
    "Echo the provided meta.task_id and meta.subagent_instance_id verbatim. "
    "See the contract rules in your system prompt."
)
