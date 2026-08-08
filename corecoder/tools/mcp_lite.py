"""MCP Lite: a minimal JSON-RPC-over-stdio MCP client with structured errors.

Optimization point #5: MCP failures must be parseable by the self-correction
loop. Instead of a bare exception or a swallowed timeout, a failed call raises
MCPToolError with a structured error_type (e.g. "MCPServerTimeout") that
ErrorClassifier maps onto RETRY_MODIFIED (longer timeout, backoff).

Phase 3.5 pre-wiring: the full MCP feature set is out of scope; this is the
client/transport seam plus the standardized timeout feedback loop.
"""

import asyncio
import json
import subprocess
import uuid


class MCPToolError(Exception):
    """Structured MCP failure, classifyable by ErrorClassifier."""

    def __init__(self, error_type: str, server: str, tool: str, message: str) -> None:
        self.error_type = error_type
        self.server = server
        self.tool = tool
        self.message = message
        super().__init__(message)


class StdioTransport:
    """JSON-RPC 2.0 over a child process's stdin/stdout (the MCP stdio transport)."""

    def __init__(self, command: list[str]) -> None:
        self._command = command
        self._proc: subprocess.Popen | None = None
        self._pending: dict[str, asyncio.Future] = {}
        self._reader_task: asyncio.Task | None = None

    async def connect(self) -> None:
        self._proc = await asyncio.to_thread(
            subprocess.Popen,
            self._command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._reader_task = asyncio.create_task(self._read_loop())

    async def send(self, msg: dict) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise MCPToolError("MCPNotConnected", "", "", "transport not connected")
        line = json.dumps(msg, ensure_ascii=False)
        await asyncio.to_thread(self._proc.stdin.write, line + "\n")
        await asyncio.to_thread(self._proc.stdin.flush)

    async def receive(self, request_id: str) -> dict:
        """Await the response matching request_id; caller applies the timeout."""
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self._pending[request_id] = fut
        try:
            return await fut
        finally:
            self._pending.pop(request_id, None)

    async def close(self) -> None:
        if self._reader_task is not None:
            self._reader_task.cancel()
        if self._proc is not None:
            await asyncio.to_thread(self._proc.terminate)

    async def _read_loop(self) -> None:
        if self._proc is None or self._proc.stdout is None:
            return
        while True:
            line = await asyncio.to_thread(self._proc.stdout.readline)
            if not line:
                for fut in self._pending.values():
                    fut.set_exception(
                        MCPToolError("MCPConnectionClosed", "", "", "server exited")
                    )
                break
            try:
                resp = json.loads(line)
            except json.JSONDecodeError:
                continue
            req_id = str(resp.get("id", ""))
            fut = self._pending.get(req_id)
            if fut is not None and not fut.done():
                fut.set_result(resp)


class MCPClientLite:
    """Minimal MCP client with a structured-timeout contract."""

    MCP_TIMEOUT = 10  # seconds

    def __init__(self, server_name: str, transport=None, timeout: float | None = None) -> None:
        self.server_name = server_name
        self._transport = transport
        self.timeout = timeout if timeout is not None else self.MCP_TIMEOUT

    async def _call_tool(self, tool_name: str, params: dict) -> dict:
        if self._transport is None:
            raise MCPToolError("MCPNotConfigured", self.server_name, tool_name, "no transport")
        msg = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": params},
        }
        try:
            await self._transport.send(msg)
            resp = await asyncio.wait_for(
                self._transport.receive(msg["id"]), timeout=self.timeout
            )
            return resp
        except asyncio.TimeoutError:
            # === 结构化错误，可被 ErrorClassifier 识别 ===
            raise MCPToolError(
                error_type="MCPServerTimeout",
                server=self.server_name,
                tool=tool_name,
                message=(
                    f"MCP Server '{self.server_name}' 响应超时 ({self.timeout}s)"
                ),
            ) from None
