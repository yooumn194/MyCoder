"""MCP call observability: one structured trace per MCP invocation.

Every call through MCPToolAdapter emits a structured MCPCallTrace (success or
failure) so the audit trail can answer "which MCP calls happened, how long did
they take, and did any fail" — the same discipline as the sandbox layer.
"""

from dataclasses import asdict, dataclass

from .logger import get_logger

logger = get_logger()


@dataclass
class MCPCallTrace:
    server: str
    tool: str
    duration_ms: float
    success: bool
    error_type: str | None = None
    request_size_bytes: int = 0
    response_size_bytes: int = 0
    sse_reconnected: bool = False
    last_event_id_used: str | None = None


def log_trace(trace: MCPCallTrace) -> None:
    if trace.success:
        logger.info("mcp_call", **asdict(trace))
    else:
        logger.warning("mcp_call_failed", **asdict(trace))
