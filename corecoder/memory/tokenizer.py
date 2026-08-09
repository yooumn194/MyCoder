"""Chinese-aware tokenization with graceful degradation.

Primary backend: jieba (word segmentation). When jieba is not installed we fall
back to a zero-dependency tokenizer that emits ASCII alnum words plus CJK
bigrams — the standard degraded mode that still gives *word-level* (≥2 char)
matching instead of single characters.

Critical contract: the fallback must produce tokens that the FTS5 'ascii'
tokenizer re-tokenizes *identically*, so a token we emit is exactly one FTS5
term. FTS5 'ascii' keeps ASCII alphanumerics and (in the SQLite builds used
here) CJK characters as token characters; everything else is a separator.
The same tokenizer is used for both indexing and querying, which keeps the
token vocabulary consistent (implicit AND in FTS5 MATCH depends on it).
"""

from __future__ import annotations

import re

# CJK Unified Ideographs block (the spans jieba handles as Chinese).
_CJK_RUN = re.compile(r"[一-鿿]+")
_ASCII_RUN = re.compile(r"[A-Za-z0-9]+")
# Short CJK runs are also emitted verbatim so exact multi-char phrases keep
# phrase-level power alongside their bigrams.
_MAX_FULL_RUN = 8

_jieba = None
try:  # optional dependency — the system degrades to bigrams without it
    import jieba  # type: ignore

    _jieba = jieba
except ImportError:  # pragma: no cover - exercised via monkeypatch
    _jieba = None


def jieba_available() -> bool:
    return _jieba is not None


def tokenize_chinese(text: str) -> str:
    """Return a space-joined token stream for FTS5 indexing / matching.

    Empty or punctuation-only text yields "" (never matches anything).
    """
    if not text:
        return ""
    if _jieba is not None:
        return " ".join(t for t in _jieba.cut(text) if t.strip())
    return _tokenize_fallback(text)


def _tokenize_fallback(text: str) -> str:
    tokens: list[str] = []
    for m in _ASCII_RUN.finditer(text):
        tokens.append(m.group(0).lower())
    for m in _CJK_RUN.finditer(text):
        run = m.group(0)
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
            if len(run) <= _MAX_FULL_RUN:
                tokens.append(run)
    return " ".join(tokens)


def build_match_query(text: str) -> str:
    """Build a safe FTS5 MATCH expression from a tokenized query string.

    Each token is double-quoted so FTS5 operators (AND / OR / NOT / NEAR /
    column:...) can never be interpreted as syntax — a user searching "not"
    must not raise a MATCH parse error.
    """
    tokens = (text or "").split()
    return " ".join(f'"{t}"' for t in tokens)
