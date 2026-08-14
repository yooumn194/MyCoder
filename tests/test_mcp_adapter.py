"""Tests for MCPToolAdapter: schema conversion, content formatting, error mapping."""

import pytest

from mycoder.mcp.adapter import MCPToolAdapter
from mycoder.mcp.errors import MCPRemoteError, MCPToolError


class _FakeClient:
    def __init__(self, content=None, remote_error=None):
        self._content = content if content is not None else [{"type": "text", "text": "hello"}]
        self._remote_error = remote_error
        self.calls = []

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if self._remote_error is not None:
            raise self._remote_error
        return {"content": self._content}


def _adapter(client=None, schema=None, **kw):
    client = client or _FakeClient()
    return MCPToolAdapter(
        client,
        "fs",
        {
            "name": "read_file",
            "description": "read a file",
            "inputSchema": schema or {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        },
    )


def test_adapter_execute_returns_string():
    a = _adapter()
    r = a.execute(path="/workspace/x.txt")
    assert r == "hello"  # text content -> plain string
    assert a._client.calls == [("read_file", {"path": "/workspace/x.txt"})]


def test_adapter_image_content_placeholder():
    client = _FakeClient(content=[
        {"type": "image", "mimeType": "image/png", "data": "AAAA"},
        {"type": "text", "text": "done"},
    ])
    a = _adapter(client)
    r = a.execute(path="/x")
    assert "[Image: image/png base64 data omitted]" in r
    assert "done" in r


def test_adapter_mcp_error_code_mapping():
    cases = {
        -32600: "MCPInvalidRequest",
        -32601: "MCPMethodNotFound",
        -32602: "MCPInvalidParams",
        -32603: "MCPInternalError",
        -32000: "MCPServerError",
        999: "MCPUnknownError",
    }
    for code, expected in cases.items():
        client = _FakeClient(remote_error=MCPRemoteError(code, "boom"))
        a = _adapter(client)
        with pytest.raises(MCPToolError) as exc:
            a.execute(path="/x")
        assert exc.value.error_type == expected, code


def test_adapter_schema_conversion():
    a = _adapter(schema={
        "type": "object",
        "properties": {"path": {"type": "string"}, "recursive": {"type": "boolean"}},
        "required": ["path"],
    })
    assert a.parameters == {
        "type": "object",
        "properties": {"path": {"type": "string"}, "recursive": {"type": "boolean"}},
        "required": ["path"],
    }
    assert a.name == "mcp_fs_read_file"
    assert a.description == "read a file"


def test_adapter_transparent_to_planning(monkeypatch):
    """mcp_* tools are ordinary mutation tools for the planning guard."""
    from mycoder import planner as planner_mod

    clear = planner_mod.clear_active_plan
    clear()
    monkeypatch.setenv("MYCODER_ENFORCE_PLANNING", "1")
    a = _adapter()
    assert a.name.startswith("mcp_")
    assert planner_mod.planning_guard(a.name) is not None  # blocked without a plan
    clear()
