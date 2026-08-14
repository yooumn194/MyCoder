"""Tests for the Streamable HTTP transport (Module C, MCP 2025-03-26)."""

import asyncio
import json

import pytest
from aiohttp import web

from mycoder.mcp.errors import MCPToolError
from mycoder.mcp.transport.streamable_http import StreamableHTTPTransport


class _StreamServer:
    """A fake MCP server: the POST response body IS the SSE stream."""

    def __init__(self, *, status=200, delay=0.0, respond=True, handler=None):
        self.status = status
        self.delay = delay
        self.respond = respond
        self.app = web.Application()
        self.app.router.add_post("/endpoint", handler or self._endpoint)

    async def _endpoint(self, request):
        data = await request.json()
        resp = web.StreamResponse(status=self.status, headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.respond:
            payload = json.dumps({"jsonrpc": "2.0", "id": data["id"], "result": {"echo": data.get("method")}})
            await resp.write(f"data: {payload}\n\n".encode())
        await resp.write_eof()
        return resp


async def _start(server) -> str:
    runner = web.AppRunner(server.app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return f"http://127.0.0.1:{port}/endpoint"


async def test_streamable_http_send_request():
    server = _StreamServer()
    url = await _start(server)
    t = StreamableHTTPTransport(url, name="test", timeout=5)
    await t.start()
    try:
        r = await t.send_request("tools/list", {})
        assert r["echo"] == "tools/list"
    finally:
        await t.close()
    await server.app.cleanup()


async def test_streamable_http_202_accepted():
    """202 Accepted still carries the response on the POST body (after a delay)."""
    server = _StreamServer(status=202, delay=0.2)
    url = await _start(server)
    t = StreamableHTTPTransport(url, name="test", timeout=5)
    await t.start()
    try:
        r = await t.send_request("ping", {})
        assert r["echo"] == "ping"
    finally:
        await t.close()
    await server.app.cleanup()


async def test_streamable_http_timeout():
    server = _StreamServer(respond=False)  # accepts but never answers
    url = await _start(server)
    t = StreamableHTTPTransport(url, name="test", timeout=0.3)
    await t.start()
    try:
        with pytest.raises(MCPToolError) as exc:
            await t.send_request("ping", {})
        assert exc.value.error_type == "MCPServerTimeout"
    finally:
        await t.close()
    await server.app.cleanup()


async def test_streamable_http_json_response():
    """P1-3: application/json response is parsed directly (stateless mode)."""

    async def _json_handler(request):
        data = await request.json()
        return web.json_response({"jsonrpc": "2.0", "id": data["id"], "result": {"echo": data.get("method")}})

    server = _StreamServer(handler=_json_handler)
    url = await _start(server)
    t = StreamableHTTPTransport(url, name="test", timeout=5)
    await t.start()
    try:
        r = await t.send_request("ping", {})
        assert r["echo"] == "ping"
    finally:
        await t.close()
    await server.app.cleanup()


async def test_streamable_http_response_mode_auto():
    """auto mode routes by Content-Type: JSON body -> direct parse, SSE -> stream."""
    # the default _StreamServer returns text/event-stream -> SSE path still works
    server = _StreamServer()
    url = await _start(server)
    t = StreamableHTTPTransport(url, name="test", timeout=5, response_mode="auto")
    await t.start()
    try:
        r = await t.send_request("tools/list", {})
        assert r["echo"] == "tools/list"
    finally:
        await t.close()
    await server.app.cleanup()


async def test_streamable_http_http_error():
    async def _bad(request):
        return web.Response(status=500)

    server = _StreamServer(handler=_bad)
    url = await _start(server)
    t = StreamableHTTPTransport(url, name="test", timeout=5)
    await t.start()
    try:
        with pytest.raises(MCPToolError) as exc:
            await t.send_request("ping", {})
        assert exc.value.error_type == "MCPHTTPError"
    finally:
        await t.close()
    await server.app.cleanup()


def test_streamable_http_and_sse_coexist():
    """Backward compat: the SSE transport is untouched by this new option."""
    from mycoder.mcp.transport import SSETransport, StreamableHTTPTransport, StdioTransport

    assert StreamableHTTPTransport is not None
    assert SSETransport is not None
    assert StdioTransport is not None
