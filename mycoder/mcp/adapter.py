"""MCPToolAdapter: bridge one MCP server tool onto the MyCoder Tool contract.

The Agent loop only sees a plain Tool.execute(**kwargs) -> str. All MCP
specifics — JSON Schema conversion, Content[] formatting, error-code mapping —
are digested here. Errors propagate as structured MCPToolError so the Phase 3
self-correction loop can classify them (escalate / retry / fail-fast).
"""

import time

from ..tools.base import Tool
from .client import MCPClient
from .errors import MCPRemoteError, MCPToolError
from .observability import MCPCallTrace, log_trace
from .runtime import run_in_loop

_MCP_ERROR_CODE_MAP = {
    -32600: "MCPInvalidRequest",  # Invalid Request -> escalate_user
    -32601: "MCPMethodNotFound",  # Method not found -> fail_fast
    -32602: "MCPInvalidParams",   # Invalid params -> escalate_user
    -32603: "MCPInternalError",   # Internal error -> retry_same
    -32000: "MCPServerError",     # Server-defined error -> alt_method
}


class MCPToolAdapter(Tool):
    name = "mcp_unset"
    description = ""
    parameters = {}
    idempotent = False  # MCP tool semantics are unknown — never auto-retry / dedup

    def __init__(
        self,
        client: MCPClient,
        server_name: str,
        mcp_tool_schema: dict,
        security=None,
    ) -> None:
        self._client = client
        self._server_name = server_name
        self._mcp_name = mcp_tool_schema.get("name", "")
        # P0 security: param regex validators (param_validators in the server
        # config) block path traversal etc. BEFORE the call is dispatched.
        self._security = security
        self.name = f"mcp_{server_name}_{self._mcp_name}"
        base = mcp_tool_schema.get("description", "")
        # intent-aware descriptions for LSP tools (cognitive alignment)
        if server_name == "lsp":
            from .lsp_metadata import describe_lsp_tool

            self.description = describe_lsp_tool(self._mcp_name, base)
        else:
            self.description = base
        self.parameters = self._convert_schema(mcp_tool_schema.get("inputSchema", {}))

    # --------------------------------------------------------------- execute

    def execute(self, **kwargs) -> str:
        """Synchronous Tool contract; the async MCP call runs on the shared loop.

        The transport's background I/O tasks live on that loop, so the whole
        call must happen there too. Parameter regex validation runs first (a
        violation is a security rejection, not a remote call). Errors propagate
        as MCPToolError (structured, classifyable) rather than swallowed.
        """
        if self._security is not None:
            reject = self._security.validate_params(
                self._server_name, self._mcp_name, kwargs
            )
            if reject:
                raise MCPToolError(
                    error_type="MCPInvalidParams",
                    server=self._server_name,
                    tool=self._mcp_name,
                    message=reject,
                )
        return run_in_loop(self._call(kwargs))

    async def _call(self, kwargs: dict) -> str:
        start = time.monotonic()
        try:
            result = await self._client.call_tool(self._mcp_name, kwargs)
            text = self._format_content(result.get("content", []))
            log_trace(
                MCPCallTrace(
                    server=self._server_name,
                    tool=self._mcp_name,
                    duration_ms=(time.monotonic() - start) * 1000,
                    success=True,
                    request_size_bytes=len(repr(kwargs)),
                    response_size_bytes=len(text),
                )
            )
            return text
        except MCPRemoteError as e:
            error_type = self._map_mcp_error_code(e.code)
            raise MCPToolError(
                error_type=error_type,
                server=self._server_name,
                tool=self._mcp_name,
                message=e.message,
            ) from e
        except MCPToolError:
            raise
        except Exception as e:  # noqa: BLE001 - wrap unknown failures
            raise MCPToolError(
                error_type="MCPUnknownError",
                server=self._server_name,
                tool=self._mcp_name,
                message=str(e),
            ) from e

    # -------------------------------------------------------------- helpers

    @staticmethod
    def _convert_schema(mcp_schema: dict) -> dict:
        """MCP inputSchema (JSON Schema) -> MyCoder Tool parameters.

        Both are JSON-schema flavoured; MyCoder expects
        {type, properties, required}. Pass the properties through and copy the
        required list (JSON Schema places it at the top level, which is what
        MyCoder's Tool.parameters also uses).
        """
        converted = {"type": "object", "properties": {}, "required": []}
        properties = mcp_schema.get("properties", {})
        if isinstance(properties, dict):
            converted["properties"] = properties
        converted["required"] = list(mcp_schema.get("required", []))
        return converted

    @staticmethod
    def _format_content(content_list: list[dict]) -> str:
        """MCP Content[] -> plain text (images/resources become placeholders)."""
        parts: list[str] = []
        for item in content_list:
            match item.get("type"):
                case "text":
                    parts.append(item.get("text", ""))
                case "image":
                    parts.append(f"[Image: {item.get('mimeType', 'unknown')} base64 data omitted]")
                case "resource":
                    res = item.get("resource", {})
                    parts.append(
                        f"[Resource: {res.get('uri', '?')} ({res.get('mimeType', '?')})]"
                    )
                case _:
                    parts.append(str(item))
        return "\n".join(parts) if parts else "(empty response)"

    @staticmethod
    def _map_mcp_error_code(code: int) -> str:
        """MCP JSON-RPC error code -> ErrorClassifier error_type."""
        return _MCP_ERROR_CODE_MAP.get(code, "MCPUnknownError")
