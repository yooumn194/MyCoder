"""Production MCP client (Phase 3.5).

Protocol isolation: the Agent loop only ever sees a MyCoder Tool; all MCP
specifics (JSON-RPC framing, SSE reconnect, capability negotiation, Content
conversion, error-code mapping) are digested inside this package.
"""

from .adapter import MCPToolAdapter
from .client import MCPClient
from .errors import MCPRemoteError, MCPStartupError, MCPToolError, MCPTransportError
from .registry import load_mcp_tools, shutdown_mcp_tools
from .security import MCPSecurityPolicy

__all__ = [
    "MCPClient",
    "MCPRemoteError",
    "MCPStartupError",
    "MCPSecurityPolicy",
    "MCPToolAdapter",
    "MCPToolError",
    "MCPTransportError",
    "load_mcp_tools",
    "shutdown_mcp_tools",
]
