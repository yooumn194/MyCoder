"""Subagent Result Contract (RFC v1.0.1, frozen).

The contract is the envelope every subagent's output must conform to, so the
orchestrator can route on status + error.category deterministically.
"""

from .prompts import SUBAGENT_CONTRACT_PROMPT
from .subagent_result import (
    CATEGORY_RETRYABLE,
    SCHEMA_VERSION,
    MAX_ARTIFACTS,
    MAX_SUMMARY_CHARS,
    ContractValidationError,
    SubagentResultValidator,
    category_to_strategy,
    migrate_v0_1_to_v1_0,
    new_instance_id,
    parse_result,
)

__all__ = [
    "CATEGORY_RETRYABLE",
    "SCHEMA_VERSION",
    "MAX_ARTIFACTS",
    "MAX_SUMMARY_CHARS",
    "SUBAGENT_CONTRACT_PROMPT",
    "ContractValidationError",
    "SubagentResultValidator",
    "category_to_strategy",
    "migrate_v0_1_to_v1_0",
    "new_instance_id",
    "parse_result",
]
