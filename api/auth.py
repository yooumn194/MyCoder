"""API-key authentication and tenant-safe resource identifiers."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
from dataclasses import dataclass

from fastapi import Header, HTTPException

from mycoder.sandbox.logger import get_logger

logger = get_logger("mycoder.api.audit")
_TENANT_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


@dataclass(frozen=True)
class Principal:
    tenant_id: str
    key_id: str


def _configured_keys() -> dict[str, str]:
    """Read ``MYCODER_API_KEYS`` as JSON or ``tenant=secret,...``."""
    raw = os.getenv("MYCODER_API_KEYS", "").strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
        if isinstance(value, dict):
            return {
                str(tenant): str(secret)
                for tenant, secret in value.items()
                if _TENANT_RE.fullmatch(str(tenant)) and str(secret)
            }
    except json.JSONDecodeError:
        pass
    result: dict[str, str] = {}
    for item in raw.split(","):
        tenant, separator, secret = item.partition("=")
        tenant = tenant.strip()
        if separator and _TENANT_RE.fullmatch(tenant) and secret.strip():
            result[tenant] = secret.strip()
    return result


def get_principal(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> Principal:
    keys = _configured_keys()
    # The HTTP service is secure by default. Trusted loopback-only development
    # may opt out explicitly with MYCODER_REQUIRE_AUTH=false; absence of the
    # variable must never turn an accidentally exposed API into an open agent.
    require_auth = os.getenv("MYCODER_REQUIRE_AUTH", "true").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    if not keys and not require_auth:
        return Principal(tenant_id="local", key_id="local-dev")
    if not keys:
        logger.error("api_audit", action="authenticate", outcome="error", reason="no_keys_configured")
        raise HTTPException(status_code=503, detail="API authentication is required but no keys are configured")
    candidate = x_api_key
    if not candidate and authorization and authorization.lower().startswith("bearer "):
        candidate = authorization[7:].strip()
    if not candidate:
        logger.warning("api_audit", action="authenticate", outcome="denied", reason="missing_key")
        raise HTTPException(status_code=401, detail="missing API key")
    for tenant_id, secret in keys.items():
        if hmac.compare_digest(candidate, secret):
            logger.info(
                "api_audit", action="authenticate", outcome="allowed",
                tenant_id=tenant_id, key_id=tenant_id,
            )
            return Principal(tenant_id=tenant_id, key_id=tenant_id)
    logger.warning("api_audit", action="authenticate", outcome="denied", reason="invalid_key")
    raise HTTPException(status_code=401, detail="invalid API key")


def scope_session_id(tenant_id: str, public_session_id: str) -> str:
    """Derive an opaque internal id; public ids may repeat across tenants."""
    if tenant_id == "local":
        return public_session_id
    tenant_hash = hashlib.sha256(tenant_id.encode("utf-8")).hexdigest()[:12]
    return f"{tenant_hash}-{public_session_id}"


def audit(action: str, principal: Principal, **fields) -> None:
    """Emit a structured audit event without ever including credentials."""
    logger.info(
        "api_audit", action=action, outcome=fields.pop("outcome", "allowed"),
        tenant_id=principal.tenant_id, key_id=principal.key_id, **fields,
    )
