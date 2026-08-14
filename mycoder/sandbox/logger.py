"""Audit logging for the sandbox layer.

Every security-relevant event (container start, command execution, timeout
kill, blocked command, fallback decision) is emitted as a *structured* record
through structlog. Structuring events rather than writing free-text messages
is what makes the audit trail queryable later — an audit log you can't filter
("which commands ran in the last hour, and what did they return?") is not an
audit log.

structlog is a hard dependency of the package, but the import is still guarded
so a stripped-down install degrades to stdlib logging instead of crashing.
"""

import logging
import sys

try:  # pragma: no cover - structlog is a hard dep; guard for stripped installs
    import structlog

    _HAS_STRUCTLOG = True
except ImportError:
    _HAS_STRUCTLOG = False

_configured = False


def get_logger(name: str = "mycoder.sandbox") -> logging.Logger:
    """Return a configured logger (structlog when available, else stdlib)."""
    global _configured
    if _HAS_STRUCTLOG and not _configured:
        _configured = True
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                # JSON when stderr is not a TTY (CI, log files, daemons);
                # pretty console output when attached to a terminal.
                structlog.processors.JSONRenderer()
                if not sys.stderr.isatty()
                else structlog.dev.ConsoleRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
            logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
            cache_logger_on_first_use=True,
        )
    if _HAS_STRUCTLOG:
        return structlog.get_logger(name)  # type: ignore[return-value]
    return logging.getLogger(name)  # fallback
