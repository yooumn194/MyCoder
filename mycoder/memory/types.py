"""Memory data types shared across the Phase 5 memory system.

MemoryEntry is the row model for the `memories` table; PatternMemory extends it
for the Self-Correction settlement path. The type/scope/source strings here must
stay in sync with the CHECK constraints in store._SCHEMA.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

# --- controlled vocabularies (mirrored in the SQLite CHECK constraints) ------
MEMORY_TYPES = (
    "user",      # explicit user preferences / instructions
    "feedback",  # corrections and guidance the user gave
    "project",   # project-scoped facts and conventions
    "pattern",   # recovered-from-failure PATTERNs (Self-Correction)
    "reference", # pointers to docs / external resources
    "decision",  # distilled decisions (e.g. from a finished plan)
    "fact",      # generic memory_save default
)
MEMORY_SCOPES = ("project", "global")
MEMORY_SOURCES = ("auto", "user", "confirmed")

# Lower = injected earlier in the system prompt (see memory/prompt.py).
TYPE_PRIORITY: dict[str, int] = {
    "user": 0,
    "feedback": 1,
    "project": 2,
    "pattern": 3,
    "reference": 4,
    "decision": 5,
    "fact": 6,
}

DEFAULT_CONFIDENCE = 0.5
# Cosine similarity above which save() updates an existing memory instead of
# creating a new one.
DEDUP_COSINE = 0.85
# Confidence below which decay() marks a memory deprecated_by='decayed'.
DECAY_THRESHOLD = 0.15
# Embedding dimension (bge-small-zh-v1.5 is 512; the numpy fallback matches).
EMBED_DIM = 512

_ENTRY_FIELDS = (
    "id", "content", "type", "scope", "source", "confidence",
    "access_count", "last_accessed", "created_at", "updated_at",
    "deprecated_by", "metadata",
)


def new_id() -> str:
    return uuid.uuid4().hex


@dataclass
class MemoryEntry:
    """One row of the `memories` table."""

    id: str = field(default_factory=new_id)
    content: str = ""
    type: str = "fact"
    scope: str = "project"
    source: str = "auto"
    confidence: float = DEFAULT_CONFIDENCE
    access_count: int = 0
    last_accessed: float | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    deprecated_by: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.type not in MEMORY_TYPES:
            self.type = "fact"
        if self.scope not in MEMORY_SCOPES:
            self.scope = "project"
        if self.source not in MEMORY_SOURCES:
            self.source = "auto"

    def to_metadata(self) -> dict[str, Any]:
        """Serialize to a plain dict (round-trips through from_metadata)."""
        data: dict[str, Any] = {}
        for name in _ENTRY_FIELDS:
            value = getattr(self, name)
            if value is not None:
                data[name] = value
        return data

    @classmethod
    def from_metadata(cls, **kwargs: Any) -> "MemoryEntry":
        """Build an entry from a row dict (e.g. a sqlite3.Row with 'metadata'
        already JSON-parsed into a dict)."""
        if isinstance(kwargs.get("metadata"), str):
            import json

            try:
                kwargs["metadata"] = json.loads(kwargs["metadata"])
            except (json.JSONDecodeError, TypeError):
                kwargs["metadata"] = {}
        data = {k: v for k, v in kwargs.items() if k in _ENTRY_FIELDS}
        return cls(**data)


@dataclass
class PatternMemory(MemoryEntry):
    """A recovered-failure PATTERN distilled by the Self-Correction loop."""

    trigger: str = ""
    action: str = ""
    outcome: str = ""
    success_score: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        self.type = "pattern"

    def to_text(self) -> str:
        return (
            f"[PATTERN] 触发: {self.trigger} → 对策: {self.action} "
            f"→ 结果: {self.outcome} (成功度 {self.success_score:.2f})"
        )

    def get_embed_text(self) -> str:
        """The text vectorized for this pattern (content + structured fields)."""
        return f"{self.trigger} {self.action} {self.outcome} {self.content}"
