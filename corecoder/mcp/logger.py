"""Logger for the MCP layer — shares the sandbox's structlog configuration
so all audit output is consistent (JSON in CI, pretty on a TTY)."""

from ..sandbox.logger import get_logger  # noqa: F401  (re-exported for callers)
