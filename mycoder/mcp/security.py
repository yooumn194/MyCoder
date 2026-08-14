"""MCPSecurityPolicy: tool whitelist + parameter validation.

Three defensive layers ensure MCP cannot become a sandbox bypass:
  1. per-server tool whitelist (unlisted tools are never registered);
  2. parameter regex validators (e.g. a filesystem path must stay under
     /workspace — blocks path traversal through an MCP tool);
  3. secrets arrive via token_env (environment), never via the config file.
"""

import re

from .logger import get_logger

logger = get_logger()


class MCPSecurityPolicy:
    def __init__(self, config: dict) -> None:
        security = config.get("security", {})
        self._allowed: dict[str, list[str]] = security.get("allowed_tools", {})
        self._validators: dict[str, str] = security.get("param_validators", {})

    def is_tool_allowed(self, server_name: str, tool_name: str) -> bool:
        """Whitelist check: a tool not listed for its server is never registered."""
        return tool_name in self._allowed.get(server_name, [])

    def validate_params(self, server_name: str, tool_name: str, params: dict) -> str | None:
        """Return a rejection reason, or None if the params pass validation."""
        prefix = f"{server_name}.{tool_name}"
        for key, pattern in self._validators.items():
            if not key.startswith(prefix):
                continue
            field = key.split(".")[-1]
            value = params.get(field, "")
            if not re.match(pattern, str(value)):
                logger.warning(
                    "mcp_param_blocked",
                    server=server_name,
                    tool=tool_name,
                    field=field,
                    pattern=pattern,
                )
                return f"参数 {field} 不匹配安全策略: {pattern}"
        return None
