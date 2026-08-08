"""Structured MCP error hierarchy — every failure is classifyable by the
Phase 3 self-correction loop (ErrorClassifier reads error.error_type)."""


class MCPToolError(Exception):
    """A structured MCP failure with an explicit error_type.

    error_type is the contract with ErrorClassifier: it maps onto a
    CorrectionStrategy deterministically (e.g. MCPServerTimeout ->
    RETRY_MODIFIED, MCPInvalidParams -> ESCALATE_USER).
    """

    def __init__(self, error_type: str, server: str, tool: str, message: str) -> None:
        self.error_type = error_type
        self.server = server
        self.tool = tool
        self.message = message
        super().__init__(message)


class MCPRemoteError(Exception):
    """A JSON-RPC error returned by the server (has an error code)."""

    def __init__(self, code: int, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


class MCPTransportError(MCPToolError):
    """Transport-level failure (framing, connection, discovery)."""

    def __init__(self, server: str, tool: str, message: str, *, error_type: str = "MCPTransportError") -> None:
        super().__init__(error_type, server, tool, message)


class MCPStartupError(Exception):
    """MCP server discovery failed in a way that should stop startup (block)."""
