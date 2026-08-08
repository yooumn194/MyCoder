"""SSE MCP transport: HTTP POST requests + an SSE event stream for responses.

The MCP SSE spec separates the two endpoints (patch 1): the SSE stream is
established with GET on `sse_endpoint`, and the server announces the POST
`post_endpoint` via an `event: endpoint` message. post_endpoint may be set
statically in config to skip discovery.

Reconnect (patch 3): exponential backoff, Last-Event-ID replay, and an
explicit note that MCP does not mandate replay support — silent message loss
is made visible.
"""

import asyncio
import json
import uuid

from ..errors import MCPRemoteError, MCPToolError, MCPTransportError
from ..logger import get_logger

logger = get_logger()

# aiohttp is imported lazily (in __init__) so that stdio-only users never pay
# for it — the MCP layer adds zero mandatory dependencies (see pyproject [mcp]).
# The decision rationale is documented in the README's dependency note.

RECONNECT_BASE_DELAY = 1.0
RECONNECT_MAX_DELAY = 30.0
ENDPOINT_DISCOVERY_TIMEOUT = 10.0


class SSETransport:
    """SSE transport with dual-endpoint discovery, auth and reconnect."""

    def __init__(
        self,
        sse_endpoint: str,
        post_endpoint: str | None = None,
        auth: dict | None = None,
        *,
        name: str = "sse",
        timeout: float = 30.0,
        endpoint_timeout: float = ENDPOINT_DISCOVERY_TIMEOUT,
    ) -> None:
        # lazy import: a pure-stdio install never needs aiohttp
        try:
            import aiohttp  # noqa: PLC0415 - intentional lazy dependency
        except ImportError as e:
            raise MCPTransportError(
                name,
                "connect",
                "SSE 传输需要 aiohttp。请安装: pip install 'corecoder[mcp]' 或 pip install aiohttp。",
            ) from e
        self._aiohttp = aiohttp
        self._sse_endpoint = sse_endpoint
        self._post_endpoint = post_endpoint  # None -> wait for the SSE endpoint event
        self._auth = auth or {}
        self.name = name
        self._timeout = timeout
        self._endpoint_timeout = endpoint_timeout
        self._session = None
        self._sse_connection = None
        self._last_event_id: str | None = None
        self._pending: dict[str, asyncio.Future] = {}
        self._endpoint_event: asyncio.Future | None = None
        self._pending_event: str | None = None
        self._read_task: asyncio.Task | None = None

    # ------------------------------------------------------------- lifecycle

    async def _establish(self) -> None:
        """Open the session + SSE connection (no reader, no endpoint wait)."""
        headers = {"Accept": "text/event-stream"}
        auth_type = self._auth.get("type")
        if auth_type == "bearer":
            headers["Authorization"] = f"Bearer {self._auth['token']}"
        elif auth_type == "api_key":
            headers[self._auth["header"]] = self._auth["value"]
        if self._last_event_id:
            headers["Last-Event-ID"] = self._last_event_id

        if self._session is not None:
            await self._session.close()
        self._session = self._aiohttp.ClientSession(headers=headers)
        self._sse_connection = await self._session.get(self._sse_endpoint)
        if self._sse_connection.status != 200:
            raise MCPTransportError(
                self.name, "connect",
                f"SSE connect returned HTTP {self._sse_connection.status}",
            )
        # dual-endpoint discovery: if post_endpoint is not static, wait for the
        # server to announce it via the SSE `endpoint` event.
        if self._post_endpoint is None:
            self._endpoint_event = asyncio.get_running_loop().create_future()

    async def connect(self) -> None:
        """Initial connect: establish + spawn the reader + discover the endpoint.

        The reader must be running during discovery, otherwise nothing consumes
        the stream and the endpoint event can never arrive.
        """
        await self._establish()
        if self._read_task is None or self._read_task.done():
            self._read_task = asyncio.create_task(self._sse_read_loop())
        if self._endpoint_event is not None:
            try:
                self._post_endpoint = await asyncio.wait_for(
                    self._endpoint_event, timeout=self._endpoint_timeout
                )
                logger.info(
                    "mcp_sse_post_endpoint",
                    server=self.name,
                    post_endpoint=self._post_endpoint,
                )
            except asyncio.TimeoutError:
                raise MCPTransportError(
                    self.name,
                    "connect",
                    "SSE server did not provide post_endpoint within "
                    f"{self._endpoint_timeout}s. Set post_endpoint statically "
                    "in config.",
                    error_type="MCPSSEEndpointMissing",
                )
            finally:
                self._endpoint_event = None

    async def _reconnect(self) -> None:
        """Re-establish the session + SSE connection after a disconnect.

        Does NOT spawn another read loop — the running one keeps iterating on
        the fresh self._sse_connection.
        """
        await self._establish()

    async def close(self) -> None:
        if self._read_task is not None:
            self._read_task.cancel()
        if self._sse_connection is not None:
            self._sse_connection.close()
        if self._session is not None:
            await self._session.close()
        self._sse_connection = None
        self._session = None

    async def shutdown(self) -> None:
        """Unified teardown name used by the registry."""
        await self.close()

    # -------------------------------------------------------------- requests

    async def send_request(self, method: str, params: dict) -> dict:
        if self._session is None or self._post_endpoint is None:
            raise MCPToolError(
                error_type="MCPNotConnected",
                server=self.name,
                tool=method,
                message="SSE transport not connected",
            )
        req_id = str(uuid.uuid4())
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[req_id] = fut
        payload = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
        async with self._session.post(self._post_endpoint, json=payload) as resp:
            # MCP SSE POST returns 202 Accepted; the response arrives via the
            # SSE stream. Only real client/server errors (4xx/5xx) are errors.
            if resp.status >= 400:
                self._pending.pop(req_id, None)
                raise MCPToolError(
                    error_type="MCPHTTPError",
                    server=self.name,
                    tool=method,
                    message=f"HTTP {resp.status} from {self._post_endpoint}",
                )
        try:
            return await asyncio.wait_for(fut, timeout=self._timeout)
        except asyncio.TimeoutError:
            self._pending.pop(req_id, None)
            raise MCPToolError(
                error_type="MCPServerTimeout",
                server=self.name,
                tool=method,
                message=f"MCP server '{self.name}' 响应超时 ({self._timeout}s)",
            ) from None

    async def send_notification(self, method: str, params: dict | None = None) -> None:
        if self._session is None or self._post_endpoint is None:
            return
        payload = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        async with self._session.post(self._post_endpoint, json=payload):
            pass

    # ------------------------------------------------------------- SSE loop

    async def _sse_read_loop(self) -> None:
        retry_delay = RECONNECT_BASE_DELAY
        line_buffer = b""
        while True:
            had_last_event_id = self._last_event_id is not None
            try:
                async for chunk in self._sse_connection.content:  # type: ignore[union-attr]
                    line_buffer += chunk
                    while b"\n" in line_buffer:
                        raw_line, line_buffer = line_buffer.split(b"\n", 1)
                        line = raw_line.decode("utf-8", errors="replace").strip()
                        self._handle_sse_line(line)
                # stream ended normally (server closed it) — treat as a disconnect
                raise ConnectionResetError("SSE stream closed by server")
            except (self._aiohttp.ClientError, ConnectionResetError, asyncio.TimeoutError) as e:
                logger.warning(
                    "mcp_sse_disconnected",
                    server=self.name,
                    reason=str(e),
                    retry_delay_s=retry_delay,
                    had_last_event_id=had_last_event_id,
                )
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, RECONNECT_MAX_DELAY)
                try:
                    await self._reconnect()
                    retry_delay = RECONNECT_BASE_DELAY
                    if had_last_event_id:
                        # patch 3: MCP does not mandate replay — make loss visible
                        logger.info(
                            "mcp_sse_reconnected",
                            server=self.name,
                            last_event_id=self._last_event_id,
                            note=(
                                "MCP does not mandate Last-Event-ID replay support; "
                                "if messages appear missing, this server may ignore it"
                            ),
                        )
                except Exception as exc:  # noqa: BLE001 - reconnect must not kill the loop
                    logger.error(
                        "mcp_sse_reconnect_failed",
                        server=self.name,
                        error=str(exc),
                    )

    def _handle_sse_line(self, line: str) -> None:
        """Parse one SSE field: event: / id: / data: (empty line = boundary)."""
        if line.startswith("event:"):
            self._pending_event = line[6:].strip()
        elif line.startswith("id:"):
            self._last_event_id = line[3:].strip()
        elif line.startswith("data:"):
            data = line[5:].strip()
            if getattr(self, "_pending_event", None) == "endpoint":
                self._pending_event = None
                if data.startswith("http"):
                    self._post_endpoint = data
                if self._endpoint_event is not None and not self._endpoint_event.done():
                    self._endpoint_event.set_result(data)
                return
            self._pending_event = None
            try:
                message = json.loads(data)
            except json.JSONDecodeError:
                return
            self._dispatch_response(message)
        elif line == "":
            self._pending_event = None  # event boundary

    def _dispatch_response(self, message: dict) -> None:
        msg_id = message.get("id")
        if msg_id is not None and str(msg_id) in self._pending:
            fut = self._pending.pop(str(msg_id))
            if "error" in message:
                err = message["error"]
                fut.set_exception(
                    MCPRemoteError(
                        code=err.get("code", -32000),
                        message=err.get("message", "MCP remote error"),
                    )
                )
            else:
                fut.set_result(message.get("result", {}))
