"""Tests for the SSE MCP transport: dual-endpoint, auth, reconnect (Step 2)."""

import asyncio
import json

import pytest
from aiohttp import web

from corecoder.mcp.errors import MCPToolError, MCPTransportError
from corecoder.mcp.transport.sse import SSETransport


def test_sse_import_error_is_clear(monkeypatch):
    """Missing aiohttp -> a clear install hint, not a bare ImportError."""
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "aiohttp":
            raise ImportError("No module named 'aiohttp'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    with pytest.raises(MCPTransportError) as exc:
        SSETransport(sse_endpoint="https://example.com/sse", name="x")
    assert "aiohttp" in str(exc.value)
    assert "corecoder[mcp]" in str(exc.value)


class FakeSSEServer:
    """An aiohttp MCP SSE server: GET /sse streams, POST /messages responds via the stream."""

    def __init__(
        self,
        *,
        announce_endpoint: bool = True,
        response_code: int = 202,
        disconnect_after: int | None = None,
        send_id: str | None = None,
        auth_check=None,
    ):
        self.sse_headers: list[dict] = []
        self.post_headers: list[dict] = []
        self._streams: list[web.StreamResponse] = []
        self._writes = 0
        self.announce_endpoint = announce_endpoint
        self.response_code = response_code
        self.disconnect_after = disconnect_after
        self.send_id = send_id
        self.auth_check = auth_check
        self.app = web.Application()
        self.app.router.add_get("/sse", self._sse)
        self.app.router.add_post("/messages", self._messages)

    async def _sse(self, request):
        if self.auth_check and not self.auth_check(request.headers):
            return web.Response(status=401)
        self.sse_headers.append(dict(request.headers))
        resp = web.StreamResponse(
            headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"}
        )
        await resp.prepare(request)
        self._streams.append(resp)
        if self.announce_endpoint:
            post_url = f"http://{request.host}/messages"
            await resp.write(f"event: endpoint\ndata: {post_url}\n\n".encode())
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            raise

    async def _messages(self, request):
        if self.auth_check and not self.auth_check(request.headers):
            return web.Response(status=401)
        self.post_headers.append(dict(request.headers))
        if self.response_code != 202:
            return web.Response(status=self.response_code)
        data = await request.json()
        payload = json.dumps(
            {"jsonrpc": "2.0", "id": data["id"], "result": {"method": data.get("method")}}
        )
        prefix = f"id: {self.send_id}\n" if self.send_id else ""
        self._writes += 1
        for s in self._streams:
            await s.write((prefix + f"data: {payload}\n\n").encode())
            if self.disconnect_after and self._writes >= self.disconnect_after:
                await s.write_eof()
        return web.Response(status=202)


async def _start(server: FakeSSEServer) -> str:
    runner = web.AppRunner(server.app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return f"http://127.0.0.1:{port}/sse"


@pytest.fixture()
async def sse_server():
    server = FakeSSEServer()
    url = await _start(server)
    yield server, url
    await server.app.cleanup()  # closes the runner


async def test_dynamic_post_endpoint_discovery(sse_server):
    server, url = sse_server
    t = SSETransport(url, name="test", timeout=5)
    await t.connect()
    try:
        assert t._post_endpoint.endswith("/messages")  # discovered, not static
        r = await t.send_request("ping", {})
        assert r["method"] == "ping"
        assert server.post_headers  # the POST hit the right endpoint
    finally:
        await t.close()


async def test_static_post_endpoint_skips_discovery(sse_server):
    server, url = sse_server
    post_url = url.replace("/sse", "/messages")
    t = SSETransport(url, post_endpoint=post_url, name="test", timeout=5)
    await t.connect()
    try:
        r = await t.send_request("ping", {})
        assert r["method"] == "ping"
    finally:
        await t.close()


async def test_no_endpoint_event_raises_clear_error():
    server = FakeSSEServer(announce_endpoint=False)
    url = await _start(server)
    t = SSETransport(url, name="test", endpoint_timeout=0.2)
    with pytest.raises(MCPTransportError) as exc:
        await t.connect()
    assert "post_endpoint" in str(exc.value)
    assert exc.value.error_type == "MCPSSEEndpointMissing"
    await server.app.cleanup()


async def test_auth_bearer():
    server = FakeSSEServer(auth_check=lambda h: h.get("Authorization") == "Bearer sekrit")
    url = await _start(server)
    t = SSETransport(url, auth={"type": "bearer", "token": "sekrit"}, name="test", timeout=5)
    await t.connect()
    await t.close()
    assert server.sse_headers and server.sse_headers[0].get("Authorization") == "Bearer sekrit"
    await server.app.cleanup()


async def test_auth_api_key():
    server = FakeSSEServer(auth_check=lambda h: h.get("X-API-Key") == "k123")
    url = await _start(server)
    t = SSETransport(
        url, auth={"type": "api_key", "header": "X-API-Key", "value": "k123"},
        name="test", timeout=5,
    )
    await t.connect()
    await t.close()
    assert server.sse_headers[0].get("X-API-Key") == "k123"
    await server.app.cleanup()


async def test_http_post_error_mapped():
    server = FakeSSEServer(response_code=500)
    url = await _start(server)
    t = SSETransport(url, name="test", timeout=5)
    await t.connect()
    try:
        with pytest.raises(MCPToolError) as exc:
            await t.send_request("ping", {})
        assert exc.value.error_type == "MCPHTTPError"
    finally:
        await t.close()
    await server.app.cleanup()


async def test_reconnect_with_last_event_id():
    """After the server closes the stream, the client reconnects carrying Last-Event-ID."""
    server = FakeSSEServer(disconnect_after=1, send_id="42")
    url = await _start(server)
    t = SSETransport(url, name="test", timeout=5)
    await t.connect()
    try:
        r = await t.send_request("ping", {})
        assert r["method"] == "ping"
        assert t._last_event_id == "42"  # captured from the id: line

        # wait for the background reconnect (server closed after the write)
        for _ in range(50):
            if len(server.sse_headers) >= 2:
                break
            await asyncio.sleep(0.1)
        assert len(server.sse_headers) >= 2, "client did not reconnect"
        assert server.sse_headers[1].get("Last-Event-ID") == "42"
    finally:
        await t.close()
    await server.app.cleanup()
