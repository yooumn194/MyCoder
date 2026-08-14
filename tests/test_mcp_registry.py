"""End-to-end MCP registry tests: config -> transport -> client -> security -> adapter.

Drives the full Phase 3.5 pipeline through load_mcp_tools() and the real agent
dispatch, using the fake stdio MCP server (no external dependencies).
"""

from types import SimpleNamespace

import pytest
import yaml

from mycoder import mcp as mcp_pkg
from mycoder.mcp import registry as registry_mod
from mycoder.mcp import runtime as runtime_mod

from .mcp_helpers import write_fake_server

HANGING_SERVER = r"""
import json, sys, re
def send(m):
    b = json.dumps(m).encode()
    sys.stdout.buffer.write(b"Content-Length: " + str(len(b)).encode() + b"\r\n\r\n" + b); sys.stdout.buffer.flush()
buf = b""
while True:
    c = sys.stdin.buffer.read1(4096)
    if not c: break
    buf += c
    while True:
        h = buf.find(b"\r\n\r\n")
        if h == -1: break
        m = re.search(rb"Content-Length:\s*(\d+)", buf[:h])
        if not m: break
        n = int(m.group(1)); s = h + 4
        if len(buf) < s + n: break
        msg = json.loads(buf[s:s+n]); buf = buf[s+n:]
        if msg.get("method") == "initialize":
            send({"jsonrpc":"2.0","id":msg["id"],"result":{"capabilities":{}}})
        # tools/list: never respond -> discovery times out
"""


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    registry_mod.shutdown_mcp_tools()
    runtime_mod.shutdown()


def _write_config(tmp_path, server, allowed, timeout=1.0, fail_strategy="skip"):
    cfg = {
        "servers": {"filesystem": server},
        "discovery": {"initial_timeout": timeout, "fail_strategy": fail_strategy},
        "security": {"allowed_tools": {"filesystem": allowed}},
    }
    path = tmp_path / "mcp_servers.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return str(path)


def _stdio_server(script):
    return {
        "transport": "stdio",
        "command": "python3",
        "args": [script],
        "enabled": True,
    }


def test_load_mcp_tools_registers_adapter(tmp_path):
    fake = write_fake_server(tmp_path)
    config = _write_config(tmp_path, _stdio_server(fake), allowed=["echo"])
    tools = mcp_pkg.load_mcp_tools(config)
    names = [t.name for t in tools]
    assert "mcp_filesystem_echo" in names
    adapter = next(t for t in tools if t.name == "mcp_filesystem_echo")
    assert adapter.execute(text="hi") == "echo:hi"


def test_security_whitelist_filters_tools(tmp_path):
    fake = write_fake_server(tmp_path)
    config = _write_config(tmp_path, _stdio_server(fake), allowed=[])  # nothing allowed
    tools = mcp_pkg.load_mcp_tools(config)
    assert tools == []


def test_discovery_timeout_skips_server(tmp_path):
    script = tmp_path / "hanging.py"
    script.write_text(HANGING_SERVER, encoding="utf-8")
    config = _write_config(tmp_path, _stdio_server(str(script)), allowed=["echo"], timeout=0.5)
    tools = mcp_pkg.load_mcp_tools(config)
    assert tools == []  # skipped, no exception, startup continues


def test_crashing_server_does_not_block_startup(tmp_path):
    config = _write_config(
        tmp_path,
        {"transport": "stdio", "command": "nonexistent-bin-xyz", "args": [], "enabled": True},
        allowed=["echo"],
    )
    tools = mcp_pkg.load_mcp_tools(config)
    assert tools == []  # server start failed, logged, but load succeeded


def test_agent_dispatches_mcp_tool(tmp_path):
    from mycoder.agent import Agent

    fake = write_fake_server(tmp_path)
    config = _write_config(tmp_path, _stdio_server(fake), allowed=["echo"])
    mcp_tools = mcp_pkg.load_mcp_tools(config)
    assert mcp_tools  # sanity: adapter exists

    agent = Agent(llm=SimpleNamespace(), tools=mcp_tools)
    r = agent._exec_tool(SimpleNamespace(name="mcp_filesystem_echo", arguments={"text": "agent-hi"}))
    assert r == "echo:agent-hi"


# --- P0 严重修复回归 --------------------------------------------------------

def test_make_transport_streamable_http():
    """streamable_http 配置现在能加载（此前 loader 只认 stdio/sse）。"""
    from mycoder.mcp.registry import _make_transport
    from mycoder.mcp.transport.streamable_http import StreamableHTTPTransport

    t = _make_transport(
        {"transport": "streamable_http", "endpoint": "https://mcp.example.com/sse", "name": "s"}
    )
    assert isinstance(t, StreamableHTTPTransport)
    with pytest.raises(Exception):
        _make_transport({"transport": "streamable_http"})  # 缺 endpoint


def test_param_validator_blocks_unsafe_value():
    """param_validators 在调用前拦截不安全参数（此前是死代码）。"""
    from mycoder.mcp.adapter import MCPToolAdapter
    from mycoder.mcp.errors import MCPToolError
    from mycoder.mcp.security import MCPSecurityPolicy

    security = MCPSecurityPolicy(
        {
            "security": {
                "allowed_tools": {"fs": ["read_file"]},
                "param_validators": {"fs.read_file.path": r"^/workspace/.*"},
            }
        }
    )
    adapter = MCPToolAdapter(
        SimpleNamespace(),
        "fs",
        {"name": "read_file", "description": "read", "inputSchema": {"type": "object", "properties": {}}},
        security=security,
    )
    with pytest.raises(MCPToolError) as ei:
        adapter.execute(path="../etc/passwd")
    assert "不匹配安全策略" in ei.value.message


def test_fail_strategy_block_propagates(tmp_path):
    """fail_strategy=block 的 server 发现失败必须阻断启动（此前被吞掉）。"""
    from mycoder.mcp.errors import MCPStartupError

    script = tmp_path / "hanging.py"
    script.write_text(HANGING_SERVER, encoding="utf-8")
    config = _write_config(
        tmp_path, _stdio_server(str(script)), allowed=["echo"], timeout=0.3, fail_strategy="block"
    )
    with pytest.raises(MCPStartupError):
        mcp_pkg.load_mcp_tools(config)


# --- 3.5.2: startup discovery hint -----------------------------------------

def _patched_logger(monkeypatch):
    from unittest import mock

    from mycoder.mcp import registry as reg

    logger = mock.Mock()
    monkeypatch.setattr(reg, "logger", logger)
    return logger


def test_hint_when_all_servers_disabled(tmp_path, monkeypatch):
    """INFO hint fires when servers exist but all are disabled (the discoverable trap)."""
    logger = _patched_logger(monkeypatch)
    cfg = tmp_path / "mcp.yaml"
    cfg.write_text(
        yaml.dump({"servers": {"filesystem": {"enabled": False}, "github": {"enabled": False}}}),
        encoding="utf-8",
    )
    tools = mcp_pkg.load_mcp_tools(cfg)
    assert tools == []
    hint = [c for c in logger.info.call_args_list if c.args and c.args[0] == "mcp_discovery_hint"]
    assert hint
    assert len(hint[0].kwargs["disabled_servers"]) == 2
    assert str(cfg) in str(hint[0].kwargs["message"])  # tells the user where to edit


def test_no_hint_when_config_empty(tmp_path, monkeypatch):
    logger = _patched_logger(monkeypatch)
    cfg = tmp_path / "mcp.yaml"
    cfg.write_text(yaml.dump({"servers": {}}), encoding="utf-8")
    mcp_pkg.load_mcp_tools(cfg)
    assert not any(c.args and c.args[0] == "mcp_discovery_hint" for c in logger.info.call_args_list)


def test_partial_enabled_uses_debug(tmp_path, monkeypatch):
    """Partially enabled: DEBUG hint, no INFO noise."""
    logger = _patched_logger(monkeypatch)
    cfg = tmp_path / "mcp.yaml"
    cfg.write_text(
        yaml.dump({
            "servers": {
                "filesystem": {"transport": "stdio", "command": "nonexistent-bin", "args": [], "enabled": True},
                "github": {"enabled": False},
            }
        }),
        encoding="utf-8",
    )
    mcp_pkg.load_mcp_tools(cfg)  # enabled server crashes -> skipped, but hint still fires
    assert not any(c.args and c.args[0] == "mcp_discovery_hint" for c in logger.info.call_args_list)
    partial = [c for c in logger.debug.call_args_list if c.args and c.args[0] == "mcp_discovery_partial"]
    assert partial
    assert len(partial[0].kwargs["disabled"]) == 1
