"""Tests for MCPClient: handshake, tools cache, discovery timeout (Step 3)."""

import asyncio

from mycoder.mcp.client import MCPClient


class _FakeTransport:
    def __init__(self, tools=None, hang_list=False, name="fake"):
        self.requests: list[tuple[str, dict]] = []
        self.notifications: list[str] = []
        self._tools = tools or []
        self._hang = hang_list
        self.name = name

    async def send_request(self, method, params):
        self.requests.append((method, params))
        if method == "initialize":
            return {"capabilities": {"resources/subscribe": True}, "protocolVersion": "2024-11-05"}
        if method == "tools/list":
            if self._hang:
                await asyncio.sleep(100)
            return {"tools": self._tools}
        if method == "tools/call":
            return {"content": [{"type": "text", "text": "ok"}]}
        return {}

    async def send_notification(self, method, params=None):
        self.notifications.append(method)

    async def shutdown(self, graceful=True):
        pass


def _tools():
    return [
        {"name": "read_file", "description": "read", "inputSchema": {"type": "object", "properties": {}}}
    ]


async def test_initialize_handshake():
    t = _FakeTransport()
    c = MCPClient(t, "fs")
    await c.initialize()
    assert ("initialize", {"protocolVersion": "2024-11-05", "capabilities": {"roots": {"listChanged": True}}, "clientInfo": {"name": "mycoder", "version": "0.4.0"}}) in t.requests
    assert "notifications/initialized" in t.notifications
    assert c.supports("resources/subscribe") is True
    assert c.supports("nope") is False


async def test_list_tools_cached():
    t = _FakeTransport(tools=_tools())
    c = MCPClient(t, "fs")
    first = await c.list_tools()
    second = await c.list_tools()
    assert first == _tools()
    assert second == first
    assert sum(1 for m, _ in t.requests if m == "tools/list") == 1  # cached
    # force refresh issues a second request
    await c.list_tools(force_refresh=True)
    assert sum(1 for m, _ in t.requests if m == "tools/list") == 2


async def test_list_tools_timeout_marks_degraded():
    t = _FakeTransport(tools=_tools(), hang_list=True)
    c = MCPClient(t, "fs", fail_strategy="skip")
    tools = await c.list_tools(timeout=0.1)
    assert tools == []
    assert c.is_degraded is True


async def test_call_tool():
    t = _FakeTransport()
    c = MCPClient(t, "fs")
    r = await c.call_tool("read_file", {"path": "/x"})
    assert ("tools/call", {"name": "read_file", "arguments": {"path": "/x"}}) in t.requests
    assert r["content"][0]["text"] == "ok"


async def test_list_tools_recovers_after_success():
    t = _FakeTransport(tools=_tools(), hang_list=True)
    c = MCPClient(t, "fs")
    await c.list_tools(timeout=0.1)
    assert c.is_degraded
    # next successful call clears degraded
    t._hang = False
    await c.list_tools(force_refresh=True, timeout=1)
    assert c.is_degraded is False
