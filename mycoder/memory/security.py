"""Sensitive-information filtering applied before a memory is persisted.

Runs in store.save() so secrets never reach the on-disk databases. The
redaction is a best-effort regex sweep — it is NOT a substitute for holding
secrets out of the pipeline in the first place.
"""

from __future__ import annotations

import re

# (compiled pattern, replacement) — order matters: PEM blocks are long and
# greedy, so they run last after the short token-style patterns have already
# shrunk the text.
SENSITIVE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # api_key / api-key / api key / secret / password / token  :  value
    (
        re.compile(
            r"(?i)(api[_-]?key|secret|password|token)\s*[:=]\s*['\"]?\S+"
        ),
        r"\1=[REDACTED]",
    ),
    # OpenAI-style keys (sk- followed by a long base62 body)
    (re.compile(r"sk-[a-zA-Z0-9]{20,}"), "[REDACTED:OPENAI_KEY]"),
    # Bearer credentials (Authorization: Bearer ...)
    (re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*"), "[REDACTED:BEARER]"),
    # PEM private key blocks (DOTALL so the body spanning lines matches)
    (
        re.compile(
            r"-----BEGIN .*? PRIVATE KEY-----.*?-----END .*? PRIVATE KEY-----",
            re.DOTALL,
        ),
        "[REDACTED:PRIVATE_KEY]",
    ),
]

# Patterns that mark the whole value as high-risk and would never carry
# useful retrieval signal even redacted.
_ENTIRELY_SENSITIVE = re.compile(
    r"^(api[_-]?key|secret|password|authorization|token)$", re.IGNORECASE
)


def filter_sensitive(text: str) -> str:
    """Redact common secret shapes. Returns the (possibly unchanged) text."""
    if not text:
        return text
    for pattern, repl in SENSITIVE_PATTERNS:
        text = pattern.sub(repl, text)
    return text


def is_sensitive(text: str) -> bool:
    """True when the text looks like a bare secret rather than prose."""
    stripped = text.strip()
    return bool(stripped) and bool(_ENTIRELY_SENSITIVE.match(stripped))
