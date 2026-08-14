"""Load config/mcp_servers.yaml and resolve secrets from the environment.

Security rule: the YAML may reference a token by *name* (token_env) but never
contains the secret itself — it is read from os.environ at load time. A
missing token is a clear startup error, not a silent misconfiguration.
"""

import os
from pathlib import Path

import yaml

from .errors import MCPStartupError

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent / "config" / "mcp_servers.yaml"
)


def _resolve_auth(auth: dict | None) -> dict:
    if not auth:
        return {}
    auth = dict(auth)
    token_env = auth.pop("token_env", None)
    if token_env:
        token = os.getenv(token_env)
        if token is None:
            raise MCPStartupError(
                f"MCP server 需要环境变量 {token_env} 提供 token，但未设置。"
                f"请 export {token_env}=... 后重试。"
            )
        auth["token"] = token
    return auth


def load_mcp_config(config_path: Path | str | None = None) -> dict:
    """Load the MCP servers YAML. Returns empty config when the file is absent."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        return {"servers": {}, "security": {}, "discovery": {}}
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    data.setdefault("servers", {})
    data.setdefault("security", {})
    data.setdefault("discovery", {})
    return data


def enabled_servers(config: dict) -> list[tuple[str, dict]]:
    """(server_name, server_config) for every enabled server, auth resolved."""
    servers: list[tuple[str, dict]] = []
    for name, cfg in config.get("servers", {}).items():
        if not cfg.get("enabled"):
            continue
        server_cfg = dict(cfg)
        server_cfg["name"] = name
        try:
            server_cfg["auth"] = _resolve_auth(cfg.get("auth"))
        except MCPStartupError:
            raise
        servers.append((name, server_cfg))
    return servers
