"""MCPClient: session management + capability negotiation for one server.

Lifecycle: initialize -> tools/list (cached) -> call_tool -> shutdown.
tools/list is bounded by a discovery timeout (patch 2): a slow or dead server
marks the client degraded instead of blocking agent startup.
"""

import asyncio

from .logger import get_logger

logger = get_logger()

PROTOCOL_VERSION = "2024-11-05"


class MCPClient:
    """Client for a single MCP server, transport-agnostic."""

    def __init__(
        self,
        transport,
        server_name: str,
        *,
        fail_strategy: str = "skip",
        discovery_timeout: float = 10.0,
    ) -> None:
        self._transport = transport
        self.server_name = server_name
        self._fail_strategy = fail_strategy
        self._discovery_timeout = discovery_timeout
        self._capabilities: dict = {}
        self._tools_cache: list[dict] | None = None
        self._degraded = False
        self._initialized = False

    @property
    def is_degraded(self) -> bool:
        return self._degraded

    async def initialize(self) -> dict:
        """MCP handshake: declare client capabilities, store server's."""
        if self._initialized:
            return self._capabilities
        result = await self._transport.send_request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"roots": {"listChanged": True}},
                "clientInfo": {"name": "mycoder", "version": "0.4.0"},
            },
        )
        self._capabilities = result.get("capabilities", {})
        # MCP spec requires the initialized notification after the handshake
        await self._transport.send_notification("notifications/initialized", {})
        self._initialized = True
        return result

    async def list_tools(self, force_refresh: bool = False, timeout: float | None = None) -> list[dict]:
        """Cached tools/list, bounded by the discovery timeout."""
        effective = timeout if timeout is not None else self._discovery_timeout
        if self._tools_cache is not None and not force_refresh:
            return self._tools_cache
        try:
            result = await asyncio.wait_for(
                self._transport.send_request("tools/list", {}), timeout=effective
            )
            self._tools_cache = result.get("tools", [])
            self._degraded = False
            return self._tools_cache
        except asyncio.TimeoutError:
            self._degraded = True
            logger.warning(
                "mcp_tools_list_timeout",
                server=self.server_name,
                timeout=effective,
                fail_strategy=self._fail_strategy,
            )
            self._tools_cache = []
            return []

    async def call_tool(self, tool_name: str, arguments: dict) -> dict:
        return await self._transport.send_request(
            "tools/call", {"name": tool_name, "arguments": arguments}
        )

    def supports(self, capability: str) -> bool:
        """Whether the server advertised a capability (e.g. resources/subscribe)."""
        return capability in self._capabilities

    async def shutdown(self) -> None:
        await self._transport.shutdown(graceful=True)
