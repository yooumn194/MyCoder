"""ToolRegistry + MCP server registration.

Discovery is bounded (patch 2): a server that doesn't answer tools/list within
the timeout is skipped / degraded / blocks per the `fail_strategy` config,
never hanging agent startup.
"""

from pathlib import Path

from ..tools.base import Tool
from .adapter import MCPToolAdapter
from .client import MCPClient
from .config import DEFAULT_CONFIG_PATH, enabled_servers, load_mcp_config
from .errors import MCPStartupError
from .logger import get_logger
from .runtime import run_in_loop
from .security import MCPSecurityPolicy
from .transport.sse import SSETransport
from .transport.stdio import StdioTransport
from .transport.streamable_http import StreamableHTTPTransport

logger = get_logger()

# Every transport the loader started, so tests / teardown can close them.
_ACTIVE_TRANSPORTS: list = []


def _track(transport) -> None:
    _ACTIVE_TRANSPORTS.append(transport)


def shutdown_mcp_tools() -> None:
    """Close every transport started by load_mcp_tools (teardown helper).

    Each shutdown is bounded so a stuck transport can't hang process exit.
    Best-effort: stdio child processes and SSE/HTTP sessions are reaped here —
    without this the CLI leaves orphan `npx`/server processes behind.
    """
    for transport in _ACTIVE_TRANSPORTS:
        try:
            run_in_loop(transport.shutdown(), timeout=2.0)
        except Exception:  # noqa: BLE001 - best-effort teardown
            pass
    _ACTIVE_TRANSPORTS.clear()


class ToolRegistry:
    """A dynamic registry for MCP adapter tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all(self) -> list[Tool]:
        return list(self._tools.values())


def _make_transport(server_cfg: dict):
    transport_type = server_cfg.get("transport")
    name = server_cfg.get("name", "mcp")
    if transport_type == "stdio":
        return StdioTransport(
            server_cfg["command"],
            server_cfg.get("args", []),
            env=server_cfg.get("env"),
            name=name,
            timeout=float(server_cfg.get("timeout", 30)),
            warmup=bool(server_cfg.get("warmup", False)),
            warmup_timeout=float(server_cfg.get("warmup_timeout", 30)),
        )
    if transport_type == "sse":
        return SSETransport(
            server_cfg["sse_endpoint"],
            post_endpoint=server_cfg.get("post_endpoint"),
            auth=server_cfg.get("auth"),
            name=name,
        )
    if transport_type == "streamable_http":
        # SSE 传输已标记弃用并提示迁移到这里；loader 此前不认它，导致
        # 配置写 streamable_http 直接启动失败（半成品）。
        endpoint = server_cfg.get("endpoint") or server_cfg.get("url")
        if not endpoint:
            raise MCPStartupError("streamable_http 需要 endpoint 或 url 配置")
        return StreamableHTTPTransport(
            endpoint,
            auth=server_cfg.get("auth"),
            name=name,
            timeout=float(server_cfg.get("timeout", 30)),
        )
    raise MCPStartupError(f"未知 MCP transport: {transport_type!r}")


async def register_server(client, config, security, registry, *, server_name: str) -> int:
    discovery = config.get("discovery", {})
    timeout = float(discovery.get("initial_timeout", 10))
    strategy = discovery.get("fail_strategy", "skip")

    tools = await client.list_tools(timeout=timeout)

    if not tools and client.is_degraded:
        if strategy == "skip":
            logger.warning("mcp_server_skipped", server=server_name, reason="discovery timeout")
            return 0
        if strategy == "block":
            raise MCPStartupError(f"Critical server {server_name} failed discovery")
        logger.warning("mcp_server_degraded", server=server_name)  # partial

    registered = 0
    for tool_schema in tools:
        tool_name = tool_schema.get("name", "")
        if not security.is_tool_allowed(server_name, tool_name):
            logger.debug("mcp_tool_blocked", server=server_name, tool=tool_name)
            continue
        registry.register(MCPToolAdapter(client, server_name, tool_schema, security=security))
        registered += 1
    logger.info(
        "mcp_server_registered", server=server_name,
        registered=registered, total=len(tools),
    )
    return registered


def _emit_discovery_hint(config: dict, config_path) -> None:
    """Help a first-time MCP configurator who enabled nothing yet.

    INFO only when servers exist but all are disabled (the discoverable trap);
    DEBUG when some are skipped; nothing when the config is empty. Pure logging
    — zero effect on registration behaviour.
    """
    all_servers = config.get("servers", {})
    enabled = [n for n, s in all_servers.items() if s.get("enabled")]
    disabled = [n for n, s in all_servers.items() if not s.get("enabled")]
    if all_servers and not enabled:
        logger.info(
            "mcp_discovery_hint",
            message=(
                f"发现 {len(all_servers)} 个 MCP Server 配置，但均未启用。"
                f"如需启用，在 {config_path} 中将对应 server 的 enabled 设为 true。"
            ),
            disabled_servers=disabled,
        )
    elif disabled:
        logger.debug(
            "mcp_discovery_partial",
            message=f"已启用 {len(enabled)} 个 MCP Server，跳过未启用的: {disabled}",
            enabled=enabled,
            disabled=disabled,
        )


async def _load_mcp_tools_async(config: dict) -> list[Tool]:
    security = MCPSecurityPolicy(config)
    registry = ToolRegistry()
    for server_name, server_cfg in enabled_servers(config):
        transport = _make_transport(server_cfg)
        _track(transport)
        client = MCPClient(transport, server_name)
        try:
            if server_cfg.get("transport") == "stdio":
                await transport.start()
            else:
                await transport.connect()
            await client.initialize()
            await register_server(client, config, security, registry, server_name=server_name)
        except MCPStartupError as exc:
            # fail_strategy=block must actually block: a critical server failing
            # discovery propagates instead of being silently skipped. The
            # strategy lives on the TOP-LEVEL config, not the per-server block.
            strategy = (config.get("discovery") or {}).get("fail_strategy", "skip")
            if strategy == "block":
                raise
            logger.error("mcp_server_start_failed", server=server_name, error=str(exc))
        except Exception as exc:  # noqa: BLE001 - a bad server must not kill startup
            logger.error("mcp_server_start_failed", server=server_name, error=str(exc))
    return registry.all()


def load_mcp_tools(config_path=None) -> list[Tool]:
    """Synchronously load registered MCP adapters (runs on the shared MCP loop).

    The returned adapters stay bound to their live transports; keep the list
    referenced for the session.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    config = load_mcp_config(config_path)
    _emit_discovery_hint(config, path)
    return run_in_loop(_load_mcp_tools_async(config))
