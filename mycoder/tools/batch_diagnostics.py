"""BatchDiagnostics (P1-2) — coalesce repeated write_file edits into one
LSP diagnostics request instead of N round-trips.

Accumulates file paths; auto-flushes when the threshold is reached, and can be
flushed explicitly (e.g. at conversation end) so a single pending file still
gets immediate diagnostics.
"""

from typing import Callable, Optional


class BatchDiagnostics:
    def __init__(
        self,
        threshold: int = 3,
        trigger: Optional[Callable[[list[str]], None]] = None,
    ) -> None:
        self._pending: list[str] = []
        self._threshold = threshold
        self._trigger = trigger  # callable(files) -> None
        self.trigger_count = 0

    def add(self, file_path: str) -> None:
        """Register a file as needing diagnostics; auto-flush at threshold."""
        self._pending.append(file_path)
        if len(self._pending) >= self._threshold:
            self.flush()

    def flush(self) -> None:
        """Trigger diagnostics for all pending files now (immediate path)."""
        if not self._pending:
            return
        files = list(self._pending)
        self._pending.clear()
        if self._trigger is not None:
            self._trigger(files)
        self.trigger_count += 1

    @property
    def pending(self) -> list[str]:
        return list(self._pending)
