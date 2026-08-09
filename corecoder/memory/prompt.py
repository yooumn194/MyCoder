"""MemoryPromptInjector: turn retrieved memories into a bounded system-prompt
section that informs planning_guard before a todo_write.

Memories are sorted by type priority (user > feedback > project > pattern >
reference > decision > fact) and appended until the token budget is exhausted.
Token counting uses tiktoken when present and a CJK-aware character heuristic
otherwise.
"""

from __future__ import annotations

from .retriever import HybridRetriever
from .types import TYPE_PRIORITY

MAX_MEMORY_TOKENS = 2048
_SECTION_HEADER = "📚 跨会话记忆（来自历史会话，供参考）:"
_TRAILER = "以上记忆仅作参考，与当前任务冲突时以本次对话为准。"

try:  # optional: precise token counts when tiktoken is installed
    import tiktoken

    _ENCODER = tiktoken.get_encoding("cl100k_base")
except ImportError:  # pragma: no cover - graceful char-heuristic fallback
    _ENCODER = None


def estimate_tokens(text: str) -> int:
    """Token estimate for `text`. CJK chars cost more than ASCII words."""
    if not text:
        return 0
    if _ENCODER is not None:
        try:
            return len(_ENCODER.encode(text))
        except Exception:  # pragma: no cover - encoder edge cases
            pass
    ascii_chars = sum(1 for ch in text if ch.isascii())
    cjk_chars = len(text) - ascii_chars
    return (ascii_chars + 3) // 4 + cjk_chars


def format_memory_entry(entry: dict) -> str:
    """One rendered line for the prompt section."""
    tag = entry["type"]
    scope = "全局" if entry["scope"] == "global" else "项目"
    conf = entry.get("confidence", 0.0)
    return f"- [{tag}|{scope}|{conf:.2f}] {entry['content']}"


class MemoryPromptInjector:
    def __init__(self, retriever: HybridRetriever, *, max_tokens: int = MAX_MEMORY_TOKENS):
        self.retriever = retriever
        self.max_tokens = max_tokens

    def build_memory_section(self, query: str) -> str:
        """Retrieve up to 20 memories (min_conf=0.3) and render a bounded
        section. Returns "" when nothing is worth injecting."""
        if not query or not query.strip():
            return ""
        results = self.retriever.search(query, limit=20, min_conf=0.3)
        if not results:
            return ""
        results.sort(key=lambda r: TYPE_PRIORITY.get(r["type"], 99))
        lines: list[str] = []
        used = estimate_tokens(_SECTION_HEADER) + estimate_tokens(_TRAILER)
        for result in results:
            line = format_memory_entry(result)
            cost = estimate_tokens(line + "\n")
            if used + cost > self.max_tokens:
                break
            lines.append(line)
            used += cost
        if not lines:
            return ""
        return "\n".join([_SECTION_HEADER, *lines, _TRAILER, ""])
