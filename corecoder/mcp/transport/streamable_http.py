"""MCP 2025-03-26 Streamable HTTP Transport.

The defining semantic (vs the SSE transport): the RESPONSE comes back on the
POST's own response body as a `text/event-stream` — there is no separate GET
stream. A 202 Accepted means "accepted; the stream may deliver responses
later", so both 200 and 202 are read as an SSE stream from the response body.

Coexists with the SSE transport (Phase 3.5) as a transport option — nothing
about the existing SSE path changes.
"""

import asyncio
import json
import uuid

from ..errors import MCPRemoteError, MCPToolError, MCPTransportError
from ..logger import get_logger

logger = get_logger()


class StreamableHTTPTransport:
    def __init__(
        self,
        endpoint: str,
        auth: dict | None = None,
        *,
        name: str = "streamable-http",
        timeout: float = 30.0,
        response_mode: str = "auto",
    ) -> None:
        # P1-3: auto (by Content-Type) | sse (force stream) | json (stateless)
        self._response_mode = response_mode
        try:
            import aiohttp  # noqa: PLC0415 - lazy, like the SSE transport
        except ImportError as e:
            raise MCPTransportError(
                name, "connect",
                "Streamable HTTP 传输需要 aiohttp。请安装: pip install 'corecoder[mcp]'。",
            ) from e
        self._aiohttp = aiohttp
        self._endpoint = endpoint
        self._auth = auth or {}
        self.name = name
        self._timeout = timeout
        self._session = None
        self._pending: dict[str, asyncio.Future] = {}
        self._readers: set[asyncio.Task] = set()

    async def start(self) -> None:
        headers = {"Accept": "application/json, text/event-stream"}
        if self._auth.get("type") == "bearer":
            headers["Authorization"] = f"Bearer {self._auth['token']}"
        self._session = self._aiohttp.ClientSession(headers=headers)

    async def close(self) -> None:
        for reader in list(self._readers):
            reader.cancel()
        if self._session is not None:
            await self._session.close()
        self._session = None

    async def shutdown(self) -> None:
        await self.close()

    # -------------------------------------------------------------- requests

    async def send_request(self, method: str, params: dict) -> dict:
        if self._session is None:
            raise MCPToolError(
                error_type="MCPNotConnected", server=self.name, tool=method,
                message="transport not started",
            )
        req_id = str(uuid.uuid4())
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[req_id] = fut
        payload = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}

        resp = await self._session.post(self._endpoint, json=payload)
        if resp.status >= 400:
            self._pending.pop(req_id, None)
            await resp.release()
            raise MCPToolError(
                error_type="MCPHTTPError", server=self.name, tool=method,
                message=f"HTTP {resp.status} from {self._endpoint}",
            )
        content_type = resp.headers.get("Content-Type", "")
        want_json = self._response_mode == "json" or (
            self._response_mode == "auto" and "application/json" in content_type
        )
        if want_json:
            # P1-3: stateless JSON response — the body IS the JSON-RPC message
            data = await resp.json()
            await resp.release()
            self._dispatch(data)
            try:
                return await asyncio.wait_for(fut, timeout=self._timeout)
            except asyncio.TimeoutError:
                self._pending.pop(req_id, None)
                raise MCPToolError(
                    error_type="MCPServerTimeout", server=self.name, tool=method,
                    message=f"MCP server '{self.name}' 响应超时 ({self._timeout}s)",
                ) from None

        # SSE stream mode: 200 and 202 both carry the stream on the response body.
        reader = asyncio.create_task(self._read_stream(resp))
        self._readers.add(reader)
        reader.add_done_callback(self._readers.discard)

        try:
            return await asyncio.wait_for(fut, timeout=self._timeout)
        except asyncio.TimeoutError:
            self._pending.pop(req_id, None)
            reader.cancel()  # don't leak the open stream
            raise MCPToolError(
                error_type="MCPServerTimeout", server=self.name, tool=method,
                message=f"MCP server '{self.name}' 响应超时 ({self._timeout}s)",
            ) from None

    async def send_notification(self, method: str, params: dict | None = None) -> None:
        if self._session is None:
            return
        payload = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        async with self._session.post(self._endpoint, json=payload) as resp:
            if resp.status >= 400:
                raise MCPToolError(
                    error_type="MCPHTTPError", server=self.name, tool=method,
                    message=f"HTTP {resp.status}",
                )

    # ---------------------------------------------------------------- stream

    async def _read_stream(self, resp) -> None:
        """Read the POST response body as an SSE stream, dispatching frames."""
        buffer = b""
        try:
            async for chunk in resp.content.iter_any():
                buffer += chunk
                while b"\n" in buffer:
                    raw_line, buffer = buffer.split(b"\n", 1)
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if line.startswith("data:"):
                        data = line[5:].strip()
                        try:
                            message = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        self._dispatch(message)
        except (asyncio.CancelledError, self._aiohttp.ClientError, ConnectionResetError):
            pass
        finally:
            await resp.release()

    def _dispatch(self, message: dict) -> None:
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
