"""Tests for MCP Lite: structured timeout errors the correction loop can parse."""

import asyncio

import pytest

from corecoder.tools.correction import CorrectionStrategy, ErrorClassifier
from corecoder.tools.mcp_lite import MCPClientLite, MCPToolError


class _NeverResponds:
    """A transport that accepts messages but never answers — the timeout case."""

    def __init__(self):
        self.sent = []

    async def send(self, msg):
        self.sent.append(msg)

    async def receive(self, request_id):
        await asyncio.sleep(100)  # hangs until the client's wait_for fires


class _Responds:
    def __init__(self, payload):
        self._payload = payload

    async def send(self, msg):
        pass

    async def receive(self, request_id):
        return self._payload


async def test_timeout_raises_structured_mcp_error():
    client = MCPClientLite(server_name="filesystem", transport=_NeverResponds(), timeout=0.05)
    with pytest.raises(MCPToolError) as exc:
        await client._call_tool("read_file", {"path": "/x"})
    assert exc.value.error_type == "MCPServerTimeout"
    assert exc.value.server == "filesystem"
    assert "响应超时" in exc.value.message


async def test_successful_call_returns_response():
    client = MCPClientLite(server_name="filesystem", transport=_Responds({"id": "1", "result": "ok"}))
    resp = await client._call_tool("list", {})
    assert resp["result"] == "ok"


def test_mcp_timeout_classified_as_retry_modified():
    """The correction loop must recognize MCPServerTimeout -> RETRY_MODIFIED."""
    err = MCPToolError("MCPServerTimeout", "filesystem", "read_file", "timeout")
    strategy, params = ErrorClassifier.classify(err)
    assert strategy == CorrectionStrategy.RETRY_MODIFIED
    assert params["timeout_multiplier"] == 2


async def test_no_transport_is_structured_error():
    client = MCPClientLite(server_name="fs")
    with pytest.raises(MCPToolError) as exc:
        await client._call_tool("x", {})
    assert exc.value.error_type == "MCPNotConfigured"
