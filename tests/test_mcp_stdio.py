"""Tests for the production stdio MCP transport (Step 1 of Phase 3.5)."""

import asyncio
import json

import pytest

from mycoder.mcp.errors import MCPRemoteError, MCPToolError
from mycoder.mcp.transport.stdio import StdioTransport

from .mcp_helpers import write_fake_server


def _make_transport(tmp_path, **kw):
    server = write_fake_server(tmp_path)
    return StdioTransport("python3", [server], name="fake", timeout=5, **kw)


# --- P0-3: warmup -----------------------------------------------------------

async def test_lsp_warmup_success(tmp_path):
    """warmup=True completes the initialize handshake at start()."""
    t = StdioTransport("python3", [write_fake_server(tmp_path)],
                       name="fake", timeout=5, warmup=True, warmup_timeout=3)
    await t.start()
    assert t._is_warmed_up is True
    # the server is ready — a follow-up tools/list works instantly
    tools = await t.send_request("tools/list", {})
    assert tools["tools"]
    await t.shutdown(graceful=False)


async def test_lsp_warmup_timeout_graceful(tmp_path):
    """A server that ignores initialize -> warmup times out but start() survives."""
    script = tmp_path / "silent.py"
    script.write_text(
        "import sys, time\n"
        "while True:\n"
        "    time.sleep(60)\n",
        encoding="utf-8",
    )
    t = StdioTransport("python3", [str(script)], name="fake",
                       timeout=5, warmup=True, warmup_timeout=0.2)
    await t.start()  # must not raise
    assert t._is_warmed_up is False  # degraded gracefully
    await t.shutdown(graceful=False)


def test_lsp_config_has_warmup_flag():
    from mycoder.mcp.config import load_mcp_config

    config = load_mcp_config()
    lsp = config["servers"]["lsp"]
    assert lsp.get("warmup") is True


# ---------------------------------------------------------------------------
# framing: sticky / half packets (unit-level, no subprocess)
# ---------------------------------------------------------------------------

def _frame(payload: dict) -> bytes:
    body = json.dumps(payload).encode()
    return b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body


async def test_sticky_packet_handling():
    """Two complete messages arriving in one read must both be dispatched."""
    t = StdioTransport("x", name="t")
    loop = asyncio.get_running_loop()
    f1, f2 = loop.create_future(), loop.create_future()
    t._pending = {"1": f1, "2": f2}

    buf = _frame({"jsonrpc": "2.0", "id": "1", "result": {"a": 1}}) + _frame(
        {"jsonrpc": "2.0", "id": "2", "result": {"b": 2}}
    )
    rest = t._frame_buffer(buf)
    assert rest == b""
    assert f1.result() == {"a": 1}
    assert f2.result() == {"b": 2}


async def test_half_packet_handling():
    """A frame whose body is incomplete must wait for the rest, not corrupt state."""
    t = StdioTransport("x", name="t")
    fut = asyncio.get_running_loop().create_future()
    t._pending = {"1": fut}

    body = json.dumps({"jsonrpc": "2.0", "id": "1", "result": {"ok": 1}}).encode()
    frame = b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n"
    rest = t._frame_buffer(frame + body[:10])
    assert not fut.done()  # half packet — nothing dispatched yet

    rest2 = t._frame_buffer(rest + body[10:])
    assert rest2 == b""
    assert fut.result() == {"ok": 1}


async def test_remote_error_dispatched():
    """A JSON-RPC error body must surface as MCPRemoteError with the code."""
    t = StdioTransport("x", name="t")
    fut = asyncio.get_running_loop().create_future()
    t._pending = {"9": fut}
    t._frame_buffer(_frame({"jsonrpc": "2.0", "id": "9", "error": {"code": -32602, "message": "bad"}}))
    with pytest.raises(MCPRemoteError) as exc:
        fut.result()
    assert exc.value.code == -32602


# ---------------------------------------------------------------------------
# lifecycle against the fake server subprocess
# ---------------------------------------------------------------------------

async def test_handshake_and_call(tmp_path):
    t = _make_transport(tmp_path)
    await t.start()
    try:
        hs = await t.handshake()
        assert hs.get("protocolVersion") == "2024-11-05"
        tools = await t.send_request("tools/list", {})
        assert tools["tools"][0]["name"] == "echo"
        r = await t.send_request("tools/call", {"name": "echo", "arguments": {"text": "hi"}})
        assert r["content"][0]["text"] == "echo:hi"
    finally:
        await t.shutdown(graceful=False)


async def test_server_crash_during_request(tmp_path):
    """Server exiting mid-request must fail the pending future, not hang it."""
    t = _make_transport(tmp_path)
    await t.start()
    try:
        with pytest.raises(MCPToolError) as exc:
            await t.send_request("tools/call", {"name": "crash", "arguments": {}})
        assert exc.value.error_type == "MCPConnectionClosed"
    finally:
        await t.shutdown(graceful=False)


async def test_graceful_shutdown_waits(tmp_path):
    t = _make_transport(tmp_path)
    await t.start()
    await t.handshake()
    await t.shutdown(graceful=True)
    assert t._process is None  # process torn down


async def test_stderr_correlated_with_request(tmp_path, monkeypatch):
    """stderr emitted during a request carries that request_id; idle stderr is None."""
    t = _make_transport(tmp_path)
    recorded = []
    monkeypatch.setattr(t, "_emit_stderr", lambda text, req: recorded.append((text, req)))
    await t.start()

    # idle stderr: the fake server prints "server starting" at startup, before
    # any request is in flight — its request_id must be None
    await asyncio.sleep(0.2)
    idle = [r for r in recorded if r[0] == "server starting"]
    assert idle and idle[0][1] is None

    # request stderr: the 'noise' tool writes to stderr while handling a request
    await t.handshake()
    await t.send_request("tools/call", {"name": "noise", "arguments": {}})
    busy = [r for r in recorded if r[0] == "index ready"]
    assert busy and busy[0][1] is not None  # correlated to the active request
    await t.shutdown(graceful=False)


async def test_error_returns_structured_mcp_error(tmp_path):
    """A server JSON-RPC error surfaces as MCPRemoteError with its code (the
    adapter then maps the code onto an error_type for the correction loop)."""
    t = _make_transport(tmp_path)
    await t.start()
    try:
        with pytest.raises(MCPRemoteError) as exc:
            await t.send_request("tools/call", {"name": "err", "arguments": {}})
        assert exc.value.code == -32602
    finally:
        await t.shutdown(graceful=False)


async def test_stdio_works_without_aiohttp(monkeypatch, tmp_path):
    """A pure-stdio install never needs aiohttp (lazy SSE dependency)."""
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "aiohttp":
            raise ImportError("No module named 'aiohttp'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    t = _make_transport(tmp_path)
    await t.start()
    try:
        hs = await t.handshake()
        assert hs.get("protocolVersion") == "2024-11-05"
    finally:
        await t.shutdown(graceful=False)
