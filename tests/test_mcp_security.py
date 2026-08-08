"""Tests for MCP security policy + config loading (Step 5)."""

import pytest

from corecoder.mcp.config import enabled_servers, load_mcp_config
from corecoder.mcp.errors import MCPStartupError
from corecoder.mcp.security import MCPSecurityPolicy

_CONFIG = """\
servers:
  filesystem:
    transport: stdio
    command: npx
    args: ["-y", "server", "/workspace"]
    enabled: true
  jira:
    transport: stdio
    command: node
    args: ["index.js"]
    enabled: true
  github:
    transport: sse
    sse_endpoint: https://x.example/sse
    post_endpoint: null
    auth:
      type: bearer
      token_env: TEST_MCP_TOKEN
    enabled: true

discovery:
  initial_timeout: 10
  fail_strategy: skip

security:
  allowed_tools:
    filesystem: ["read_file"]
  param_validators:
    filesystem.read_file.path: "^/workspace/.*"
"""


def _write_config(tmp_path, content=_CONFIG):
    path = tmp_path / "mcp_servers.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_token_loaded_from_env_not_config(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_MCP_TOKEN", "sekrit-token")
    config = load_mcp_config(_write_config(tmp_path))
    servers = dict(enabled_servers(config))
    assert servers["github"]["auth"]["token"] == "sekrit-token"
    # the secret never lives in the config file itself
    assert "sekrit-token" not in _CONFIG


def test_missing_env_var_raises_clear_error(tmp_path, monkeypatch):
    monkeypatch.delenv("TEST_MCP_TOKEN", raising=False)
    config = load_mcp_config(_write_config(tmp_path))
    with pytest.raises(MCPStartupError) as exc:
        enabled_servers(config)
    assert "TEST_MCP_TOKEN" in str(exc.value)


def test_unlisted_server_tools_blocked(tmp_path):
    config = load_mcp_config(_write_config(tmp_path))
    policy = MCPSecurityPolicy(config)
    assert policy.is_tool_allowed("filesystem", "read_file") is True
    assert policy.is_tool_allowed("filesystem", "write_file") is False  # not listed
    assert policy.is_tool_allowed("jira", "create_issue") is False  # jira unlisted


def test_path_traversal_blocked(tmp_path):
    config = load_mcp_config(_write_config(tmp_path))
    policy = MCPSecurityPolicy(config)
    rejected = policy.validate_params("filesystem", "read_file", {"path": "../../etc/passwd"})
    assert rejected and "不匹配安全策略" in rejected
    assert policy.validate_params("filesystem", "read_file", {"path": "/workspace/x.txt"}) is None


def test_config_defaults_disabled(tmp_path):
    """The shipped default YAML has every server disabled (opt-in)."""
    import corecoder.mcp.config as cfg

    path = cfg.DEFAULT_CONFIG_PATH
    assert path.exists()
    config = load_mcp_config(path)
    assert all(not s.get("enabled") for s in config["servers"].values())
    assert enabled_servers(config) == []
