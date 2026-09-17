"""Safe runtime configuration snapshots for reproducible evaluations.

The snapshot intentionally contains no API key or secret.  It is included in
evaluation manifests and compared on ``--resume`` so a result directory cannot
silently mix providers, models, tool dialects, or generation settings.
"""

from __future__ import annotations

import os
from urllib.parse import urlsplit, urlunsplit

from mycoder.config import Config
from mycoder.tool_protocol import provider_tool_capabilities


def _safe_base_url(value: str | None) -> str | None:
    """Keep only the URL origin/path; never persist query/fragment secrets."""
    if not value:
        return None
    parsed = urlsplit(str(value))
    if not parsed.scheme or not parsed.netloc:
        return str(value).split("?", 1)[0].split("#", 1)[0]
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))


def snapshot() -> dict[str, object]:
    """Return the effective, secret-free model/runtime configuration."""
    config = Config.from_env()
    return {
        "profile": os.getenv("MYCODER_PROFILE") or os.getenv("MYCODER_PROVIDER") or None,
        "provider": config.provider,
        "model": config.model,
        "base_url": _safe_base_url(config.base_url),
        # Snapshot the same provider-aware resolution used by LLM, including
        # provider-specific overrides, so resume manifests describe the wire
        # protocol that was actually sent.
        "tool_dialect": provider_tool_capabilities(
            config.provider,
            config.model,
            config.tool_dialect,
        ).dialect,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "max_context_tokens": config.max_context_tokens,
        "thinking": config.thinking,
    }
