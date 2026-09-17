"""Multi-layer context compression.

Claude Code uses a 4-layer strategy:
  1. HISTORY_SNIP   - trim old tool outputs to a one-line summary
  2. Microcompact   - LLM-powered summary of old turns (cached)
  3. CONTEXT_COLLAPSE - aggressive compression when nearing hard limit
  4. Autocompact    - periodic background compaction

MyCoder implements the same idea in 3 layers:
  Layer 1 (tool_snip)   - replace verbose tool results with truncated versions
  Layer 2 (summarize)   - LLM-powered summary of old conversation
  Layer 3 (hard_collapse) - last resort: drop everything except summary + recent
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .llm import LLM


def _approx_tokens(text: str) -> int:
    """Rough token count, roughly 3 chars per token for mixed en/zh content."""
    return max(1, len(text) // 3) if text else 0


def estimate_tokens(messages: list[dict]) -> int:
    total = 0
    for m in messages:
        if m.get("content"):
            total += _approx_tokens(m["content"])
        if m.get("tool_calls"):
            total += _approx_tokens(str(m["tool_calls"]))
    return total


class ContextManager:
    def __init__(self, max_tokens: int = 128_000, on_compressed=None):
        self.max_tokens = max_tokens
        # P1 memory closure: called as on_compressed(old_messages, summary)
        # after a summarization/collapse pass so the compressed content can be
        # demoted to the memory DB instead of dropped (see memory/compressor).
        self.on_compressed = on_compressed
        # layer thresholds (fraction of max_tokens)
        self._snip_at = int(max_tokens * 0.50)    # 50% -> snip tool outputs
        self._summarize_at = int(max_tokens * 0.70)  # 70% -> LLM summarize
        self._collapse_at = int(max_tokens * 0.90)   # 90% -> hard collapse
        # Compression accounting: cumulative tokens saved by compression, for
        # the "压缩省了多少 token / 压缩比" metric (面经).
        self._compression_count = 0
        self._tokens_before = 0
        self._tokens_after = 0

    def maybe_compress(self, messages: list[dict], llm: LLM | None = None) -> bool:
        """Apply compression layers as needed. Returns True if any compression happened."""
        before = estimate_tokens(messages)
        current = before
        compressed = False

        # Layer 1: snip verbose tool outputs
        if current > self._snip_at:
            if self._snip_tool_outputs(messages):
                compressed = True
                current = estimate_tokens(messages)

        # Layer 2: LLM-powered summarization of old turns
        if current > self._summarize_at and len(messages) > 10:
            if self._summarize_old(messages, llm, keep_recent=8):
                compressed = True
                current = estimate_tokens(messages)

        # Layer 3: hard collapse - last resort
        if current > self._collapse_at and len(messages) > 4:
            compressed = self._hard_collapse(messages, llm) or compressed

        if compressed:
            self._compression_count += 1
            self._tokens_before += before
            self._tokens_after += estimate_tokens(messages)

        return compressed

    def compression_stats(self) -> dict:
        """Cumulative compression accounting — the "压缩省了多少 / 压缩比" metric."""
        saved = self._tokens_before - self._tokens_after
        return {
            "compressions": self._compression_count,
            "tokens_before": self._tokens_before,
            "tokens_after": self._tokens_after,
            "tokens_saved": saved,
            "avg_compression_ratio": round(saved / self._tokens_before, 4)
            if self._tokens_before
            else 0.0,
            "avg_tokens_saved_per_compression": round(saved / self._compression_count, 1)
            if self._compression_count
            else 0.0,
        }

    def compact_for_action(self, messages: list[dict], keep_recent: int = 6) -> bool:
        """Deterministically compact exploration history before a forced edit.

        This path deliberately avoids another LLM summarization call. It keeps
        the original task, concise tool evidence, and dependency-safe recent
        messages while removing repeated exploratory prose.

        Thinking-mode providers (notably DeepSeek) require the *exact*
        ``reasoning_content`` emitted by every previous assistant turn to be
        replayed whenever tools are present.  Replacing such turns with a
        synthetic summary would make the next request invalid (HTTP 400), so
        leave the history intact and let the normal tool-output snipper reclaim
        space instead.
        """
        if len(messages) <= keep_recent + 2:
            return False
        # Thinking-mode providers require the exact reasoning fields to be
        # replayed. We may still reclaim oversized tool payloads without
        # removing those assistant turns; short histories remain unchanged.
        if any("reasoning_content" in message for message in messages):
            before = estimate_tokens(messages)
            changed = self._snip_tool_outputs(messages)
            if not changed:
                return False
            after = estimate_tokens(messages)
            if after >= before:
                return False
            self._compression_count += 1
            self._tokens_before += before
            self._tokens_after += after
            return True
        previous = list(messages)
        before = estimate_tokens(messages)
        split = self._safe_split(messages, keep_recent)
        old = messages[:split]
        tail = messages[split:]
        original = next(
            (
                str(message.get("content", ""))[:4000]
                for message in old
                if message.get("role") == "user" and message.get("content")
            ),
            "",
        )
        evidence: list[str] = []
        for message in old:
            if message.get("role") == "assistant" and message.get("tool_calls"):
                names = [
                    str(call.get("name") or call.get("function", {}).get("name") or "")
                    for call in message["tool_calls"]
                    if isinstance(call, dict)
                ]
                if names:
                    evidence.append("tools: " + ", ".join(name for name in names if name))
            elif message.get("role") == "tool" and message.get("content"):
                content = " ".join(str(message["content"]).split())
                evidence.append(content[:400])
        summary = "\n".join(evidence[-8:])[:3000] or "No durable tool evidence."
        messages[:] = [
            {
                "role": "user",
                "content": (
                    "[Action context compacted]\nOriginal task:\n"
                    f"{original}\n\nPrior tool evidence:\n{summary}"
                ),
            },
            {
                "role": "assistant",
                "content": "Context retained. Proceeding to the required repository mutation.",
            },
            *tail,
        ]
        after = estimate_tokens(messages)
        if after >= before:
            messages[:] = previous
            return False
        self._compression_count += 1
        self._tokens_before += before
        self._tokens_after += after
        return True

    @staticmethod
    def _snip_tool_outputs(messages: list[dict]) -> bool:
        """Layer 1: Truncate tool results over 1500 chars to their first/last lines.

        This mirrors Claude Code's HISTORY_SNIP which replaces old tool outputs
        with a one-line summary to reclaim context space.
        """
        changed = False
        for m in messages:
            if m.get("role") != "tool":
                continue
            content = m.get("content", "")
            if not isinstance(content, str):
                content = str(content or "")
            if len(content) <= 1500:
                continue
            lines = content.splitlines()
            if len(lines) > 6:
                # keep first 3 + last 3 lines
                snipped = (
                    "\n".join(lines[:3])
                    + f"\n... ({len(lines)} lines, snipped to save context) ...\n"
                    + "\n".join(lines[-3:])
                )
            else:
                # Compiler traces and JSON blobs are often one long line.
                # Keep both ends because the tail commonly contains the error.
                head, tail = content[:900], content[-450:]
                snipped = f"{head}\n... (output snipped to save context) ...\n{tail}"
            m["content"] = snipped
            changed = True
        return changed

    @staticmethod
    def _safe_split(messages: list[dict], keep_recent: int) -> int:
        """Index where the kept tail should start.

        Walk the boundary back so a 'tool' result is never separated from the
        assistant message whose tool_calls produced it - an orphaned tool
        message has no preceding tool_calls and OpenAI-compatible APIs reject it.
        """
        split = max(0, len(messages) - keep_recent)
        while split > 0 and messages[split].get("role") == "tool":
            split -= 1
        return split

    def _summarize_old(self, messages: list[dict], llm: LLM | None,
                       keep_recent: int = 8) -> bool:
        """Layer 2: Summarize old conversation, keep recent messages intact."""
        if len(messages) <= keep_recent:
            return False

        split = self._safe_split(messages, keep_recent)
        old = messages[:split]
        tail = messages[split:]

        # Do not discard provider-private chain-of-thought. DeepSeek's
        # thinking-mode tool protocol requires every historical
        # ``reasoning_content`` value to be sent back verbatim on subsequent
        # tool-bearing requests. A compacted prose summary is not equivalent.
        if any("reasoning_content" in message for message in old):
            return False

        summary = self._get_summary(old, llm)
        self._notify_compressed(old, summary)

        messages.clear()
        messages.append({
            "role": "user",
            "content": f"[Context compressed - conversation summary]\n{summary}",
        })
        messages.append({
            "role": "assistant",
            "content": "Got it, I have the context from our earlier conversation.",
        })
        messages.extend(tail)
        return True

    def _hard_collapse(self, messages: list[dict], llm: LLM | None) -> bool:
        """Layer 3: Emergency compression. Keep only last 4 messages + summary."""
        split = self._safe_split(messages, 4 if len(messages) > 4 else 2)
        tail = messages[split:]
        old = messages[:split]
        if any("reasoning_content" in message for message in old):
            return False
        summary = self._get_summary(old, llm)
        self._notify_compressed(old, summary)

        messages.clear()
        messages.append({
            "role": "user",
            "content": f"[Hard context reset]\n{summary}",
        })
        messages.append({
            "role": "assistant",
            "content": "Context restored. Continuing from where we left off.",
        })
        messages.extend(tail)
        return True

    def _notify_compressed(self, old_messages: list[dict], summary: str) -> None:
        """Fire the on_compressed hook so compressed content is demoted to the
        memory DB (P1). Best-effort — a failing hook never breaks compression."""
        if self.on_compressed is None:
            return
        try:
            self.on_compressed(old_messages, summary)
        except Exception:  # noqa: BLE001 - memory closure is optional
            pass

    def _get_summary(self, messages: list[dict], llm: LLM | None) -> str:
        """Generate summary via LLM or fallback to extraction."""
        flat = self._flatten(messages)

        if llm:
            try:
                resp = llm.chat(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Compress this conversation into a brief summary. "
                                "Preserve: file paths edited, key decisions made, "
                                "errors encountered, current task state. "
                                "Drop: verbose command output, code listings, "
                                "redundant back-and-forth."
                            ),
                        },
                        {"role": "user", "content": flat[:15000]},
                    ],
                )
                return resp.content
            except Exception:
                pass

        # fallback: extract key lines
        return self._extract_key_info(messages)

    @staticmethod
    def _flatten(messages: list[dict]) -> str:
        parts = []
        for m in messages:
            role = m.get("role", "?")
            text = m.get("content", "") or ""
            if text:
                parts.append(f"[{role}] {text[:400]}")
        return "\n".join(parts)

    @staticmethod
    def _extract_key_info(messages: list[dict]) -> str:
        """Fallback: extract file paths, errors, and decisions without LLM."""
        import re
        files_seen = set()
        errors = []

        for m in messages:
            text = m.get("content", "") or ""
            # extract file paths
            for match in re.finditer(r'[\w./\-]+\.\w{1,5}', text):
                files_seen.add(match.group())
            # extract error lines
            for line in text.splitlines():
                if "error" in line.lower():
                    errors.append(line.strip()[:150])

        parts = []
        if files_seen:
            parts.append(f"Files touched: {', '.join(sorted(files_seen)[:20])}")
        if errors:
            parts.append(f"Errors seen: {'; '.join(errors[:5])}")
        return "\n".join(parts) or "(no extractable context)"
