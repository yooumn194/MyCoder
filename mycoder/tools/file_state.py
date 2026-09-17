"""Per-agent file revision state for safe read-before-edit semantics."""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path


def _digest(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class FileStateTracker:
    """Remember the exact revision of files observed by the current agent.

    The state is deliberately scoped to a freshly built tool registry.  It is
    neither global nor persisted across sessions, so one tenant cannot grant
    another tenant permission to edit a file it never inspected.
    """

    def __init__(self) -> None:
        self._revisions: dict[str, str] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _key(path: Path) -> str:
        return str(path.resolve())

    def observe(self, path: Path, content: str) -> None:
        with self._lock:
            self._revisions[self._key(path)] = _digest(content)

    def validate(self, path: Path, content: str) -> str | None:
        """Return a model-facing error code when no current read is known."""
        with self._lock:
            expected = self._revisions.get(self._key(path))
        if expected is None:
            return "FILE_NOT_READ"
        if expected != _digest(content):
            return "STALE_FILE"
        return None

