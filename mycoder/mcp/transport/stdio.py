"""Production-grade stdio MCP transport.

Deltas over Phase 3's Lite prototype:
  * Content-Length framing that handles sticky and half packets
  * full server process lifecycle: start -> initialize handshake -> graceful
    shutdown -> crash restart on the next call
  * stderr capture correlated with the active request_id (so a server error
    is traceable to the call that triggered it)
  * concurrent requests serialized through an asyncio.Queue with request/response
    matching by id
"""

import asyncio
import json
import os
import re
import uuid

from ..errors import MCPRemoteError, MCPToolError
from ..logger import get_logger

logger = get_logger()

_CONTENT_LENGTH_RE = re.compile(rb"Content-Length:\s*(\d+)")
_STDERR_NOISE_LEVELS = ("error", "fatal", "panic")


class StdioTransport:
    """JSON-RPC 2.0 over a child process's stdin/stdout with framing."""

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        *,
        name: str = "stdio",
        timeout: float = 30.0,
        warmup: bool = False,
        warmup_timeout: float = 30.0,
    ) -> None:
        self._command = command
        self._args = list(args or [])
        self._env = {**os.environ, **(env or {})}
        self.name = name
        self._timeout = timeout
        # P0-3: warm the server (full initialize handshake) at start() so the
        # first LSP tool call has no 2-5s cold-start delay.
        self._warmup = warmup
        self._warmup_timeout = warmup_timeout
        self._is_warmed_up = False
        self._process: asyncio.subprocess.Process | None = None
        self._pending: dict[str, asyncio.Future] = {}
        self._request_queue: asyncio.Queue = asyncio.Queue()
        self._reader_task: asyncio.Task | None = None
        self._writer_task: asyncio.Task | None = None
        self._stderr_task: asyncio.Task | None = None
        # stderr <-> request correlation (patch 4)
        self._current_request_id: str | None = None
        self._request_lock = asyncio.Lock()

    # ------------------------------------------------------------- lifecycle

    async def start(self) -> None:
        self._process = await asyncio.create_subprocess_exec(
            self._command,
            *self._args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._env,
        )
        self._reader_task = asyncio.create_task(self._read_loop())
        self._writer_task = asyncio.create_task(self._write_loop())
        self._stderr_task = asyncio.create_task(self._stderr_capture())
        if self._warmup:
            self._is_warmed_up = await self._warmup_handshake()

    async def _warmup_handshake(self) -> bool:
        """Complete the initialize handshake now. Returns True on success.

        A timeout only logs and the server is retried on first use
        (fail-open for the warmup path only) — but the warmed-up flag must not
        be set, so callers know the server was NOT pre-warmed.
        """
        try:
            await asyncio.wait_for(
                self.send_request(
                    "initialize",
                    {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"roots": {"listChanged": True}},
                        "clientInfo": {"name": "mycoder", "version": "0.4.0"},
                    },
                ),
                timeout=self._warmup_timeout,
            )
            await self.send_notification("notifications/initialized", {})
            logger.info("mcp_warmup_ok", server=self.name, command=self._command)
            return True
        except asyncio.TimeoutError:
            logger.warning(
                "mcp_warmup_timeout",
                server=self.name,
                command=self._command,
                timeout=self._warmup_timeout,
            )
            return False

    async def handshake(self) -> dict:
        """MCP initialize handshake — the server must answer or we fail fast."""
        return await self.send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {"roots": {"listChanged": True}},
                "clientInfo": {"name": "mycoder", "version": "0.4.0"},
            },
        )

    async def _ensure_running(self) -> None:
        """Restart the server if it died (crash recovery on the next call)."""
        if self._process is not None and self._process.returncode is None:
            return
        await self.start()
        await self.handshake()

    # -------------------------------------------------------------- requests

    async def send_request(self, method: str, params: dict) -> dict:
        req_id = str(uuid.uuid4())
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[req_id] = fut
        msg = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}

        async with self._request_lock:
            self._current_request_id = req_id  # for stderr correlation
            await self._request_queue.put(msg)

        try:
            result = await asyncio.wait_for(fut, timeout=self._timeout)
            return result
        except asyncio.TimeoutError:
            self._pending.pop(req_id, None)
            raise MCPToolError(
                error_type="MCPServerTimeout",
                server=self.name,
                tool=method,
                message=f"MCP server '{self.name}' 响应超时 ({self._timeout}s)",
            ) from None
        finally:
            async with self._request_lock:
                if self._current_request_id == req_id:
                    self._current_request_id = None

    async def send_notification(self, method: str, params: dict | None = None) -> None:
        await self._request_queue.put(
            {"jsonrpc": "2.0", "method": method, "params": params or {}}
        )

    # ------------------------------------------------------------ I/O loops

    async def _write_loop(self) -> None:
        while True:
            msg = await self._request_queue.get()
            if msg is None:  # sentinel -> stop
                break
            if self._process is None or self._process.stdin is None:
                continue
            body = json.dumps(msg).encode("utf-8")
            frame = (
                b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
            )
            try:
                self._process.stdin.write(frame)
                await self._process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                # server died mid-write; the reader loop fails the pending futures
                break

    async def _read_loop(self) -> None:
        buffer = b""
        while True:
            if self._process is None or self._process.stdout is None:
                break
            chunk = await self._process.stdout.read(4096)
            if not chunk:
                break  # EOF -> server exited
            buffer += chunk
            buffer = self._frame_buffer(buffer)

        # server exited: fail every pending request rather than hang it
        for req_id, fut in self._pending.items():
            if not fut.done():
                fut.set_exception(
                    MCPToolError(
                        error_type="MCPConnectionClosed",
                        server=self.name,
                        tool="unknown",
                        message=f"MCP server '{self.name}' exited mid-request",
                    )
                )
        self._pending.clear()

    def _frame_buffer(self, buffer: bytes) -> bytes:
        """Parse Content-Length-framed messages, handling sticky/half packets."""
        while True:
            header_end = buffer.find(b"\r\n\r\n")
            if header_end == -1:
                return buffer  # incomplete header (half packet)
            m = _CONTENT_LENGTH_RE.search(buffer[:header_end])
            if m is None:
                buffer = buffer[header_end + 4 :]  # malformed frame — skip it
                continue
            content_length = int(m.group(1))
            body_start = header_end + 4
            if len(buffer) < body_start + content_length:
                return buffer  # half packet — wait for the rest
            body = buffer[body_start : body_start + content_length]
            buffer = buffer[body_start + content_length :]
            try:
                message = json.loads(body.decode("utf-8"))
            except json.JSONDecodeError:
                continue
            self._dispatch_response(message)

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
        elif "method" in message:
            # server -> client notification (resources/updated, etc.)
            asyncio.create_task(self._handle_notification(message))

    async def _handle_notification(self, message: dict) -> None:
        logger.debug("mcp_notification", server=self.name, method=message.get("method"))

    # ------------------------------------------------------------ stderr

    async def _stderr_capture(self) -> None:
        if self._process is None or self._process.stderr is None:
            return
        while True:
            line = await self._process.stderr.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").rstrip()
            async with self._request_lock:
                current_req = self._current_request_id
            self._emit_stderr(text, current_req)

    def _emit_stderr(self, text: str, request_id: str | None) -> None:
        is_noisy = any(k in text.lower() for k in _STDERR_NOISE_LEVELS)
        if is_noisy:
            logger.warning(
                "mcp_stderr",
                server=self.name,
                stream="stderr",
                request_id=request_id,
                line=text,
            )
        else:
            logger.debug(
                "mcp_stderr",
                server=self.name,
                stream="stderr",
                request_id=request_id,
                line=text,
            )

    # ------------------------------------------------------------- shutdown

    async def shutdown(self, graceful: bool = True) -> None:
        proc = self._process
        if proc is None:
            return
        if graceful and proc.returncode is None:
            try:
                await self.send_notification("notifications/exit", {})
                await asyncio.wait_for(proc.wait(), timeout=5)
            except (TimeoutError, Exception):
                pass
        if proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=3)
            except TimeoutError:
                proc.kill()
        for task in (self._reader_task, self._writer_task, self._stderr_task):
            if task is not None:
                task.cancel()
        self._process = None
