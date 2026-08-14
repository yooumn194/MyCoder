"""Memory compression (P1) — conversation → long-term memory closure.

context.py already implements the classic two-phase text compression:
Phase 1 truncates verbose tool outputs, Phase 2 LLM-summarizes old turns. What
was missing is the "don't discard, demote" closure — the differentiated part of
the layered strategy:

  * Phase 3 (extract_facts / settle_facts) — when the context compresses old
    turns, the key facts are extracted (fast-model, structured JSON) and
    persisted to the memory DB instead of being thrown away. The next agentic
    RAG pass (memory_search) can retrieve them back — the compressed content is
    demoted to the vector store, not dropped.
  * Loss audit (verify_retention) — an LLM judge compares the pre/post summary
    and scores how much key information survived. Recorded as a metric; never
    blocks compression.
  * Rolling update (summarize_cluster) — the memory DB itself compacts by
    merging low-value auto memories of a scope into one summary entry and
    DEMOTING the originals (deprecated_by='merged') instead of hard-deleting
    them, so nothing is lost and the store stays retrievable.
"""

from __future__ import annotations

import json
import re

from .store import MemoryStore, get_store
from .types import MEMORY_TYPES, MemoryEntry

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

_FACT_PROMPT = """\
你是关键事实提取器。从对话轮次中提取值得长期记住的事实：项目约定、技术决策、文件/模块位置、错误教训、用户偏好。排除临时工具输出、闲聊、重复内容。
严格输出 JSON：{"facts": [{"fact": "事实描述", "type": "project|decision|pattern|fact", "confidence": 0.0-1.0}]}
最多 8 条。"""

_RETENTION_PROMPT = """\
你是信息保留率评估员。对比「压缩前的原文」和「压缩后的摘要」，判断摘要保留了原文多少关键信息（事实、决策、结论、文件位置等）。
严格输出 JSON：{"retention": 0.0-1.0, "lost": "丢失的关键点，没有则写 '无'"}"""

_SUMMARY_PROMPT = """\
你是记忆合并器。把以下多条记忆合并成一条精炼摘要，保留所有仍然成立的事实与决策。严格输出 JSON：{"summary": "合并摘要（<=200字）"}"""


def _extract_json(raw: str) -> dict:
    if not raw:
        return {}
    match = _JSON_FENCE.search(raw)
    text = match.group(1).strip() if match else raw.strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def _flatten(messages: list[dict]) -> str:
    """Flatten a messages list into plain text for the extractor / judge."""
    parts = []
    for m in messages:
        role = m.get("role", "?")
        text = m.get("content", "") or ""
        if text:
            parts.append(f"[{role}] {str(text)[:400]}")
        if m.get("tool_calls"):
            parts.append(f"[assistant→tool] {str(m['tool_calls'])[:200]}")
    return "\n".join(parts) if parts else ""


class MemoryCompressor:
    """Demote compressed conversation content into the long-term memory DB.

    ``store`` defaults to the process singleton; ``llm`` is the (preferably
    cheapest-tier) model used for structured extraction / judging / merging.
    All methods are best-effort: a failure never breaks compression.
    """

    def __init__(
        self,
        store: MemoryStore | None = None,
        llm=None,
        *,
        scope: str = "project",
    ) -> None:
        self.store = store or get_store()
        self._llm = llm
        self.scope = scope

    # ------------------------------------------------------------ Phase 3

    def extract_facts(self, messages: list[dict]) -> list[dict]:
        """Return structured key-fact candidates from old turns (LLM or a
        rule fallback). Each: {fact, type, confidence}."""
        text = _flatten(messages)
        if not text.strip():
            return []
        if self._llm is not None:
            try:
                resp = self._llm.chat(
                    [
                        {"role": "system", "content": _FACT_PROMPT},
                        {"role": "user", "content": text[:6000]},
                    ],
                    response_format={"type": "json_object"},
                )
                data = _extract_json(str(getattr(resp, "content", "")))
                facts = data.get("facts") if isinstance(data.get("facts"), list) else []
                cleaned = []
                for f in facts:
                    if isinstance(f, dict) and str(f.get("fact", "")).strip():
                        cleaned.append(
                            {
                                "fact": str(f["fact"]).strip()[:300],
                                "type": str(f.get("type", "fact")),
                                "confidence": _as_float(f.get("confidence"), 0.6),
                            }
                        )
                if cleaned:
                    return cleaned
            except Exception:  # noqa: BLE001 - fall back to rules
                pass
        return self._rule_facts(messages)

    def settle_facts(self, messages: list[dict]) -> list[str]:
        """Extract facts and persist them; returns the saved memory ids."""
        ids: list[str] = []
        for f in self.extract_facts(messages):
            content = str(f.get("fact", "")).strip()
            if not content:
                continue
            ftype = str(f.get("type", "fact"))
            if ftype not in MEMORY_TYPES:
                ftype = "fact"
            confidence = _clamp(f.get("confidence", 0.6))
            entry = MemoryEntry(
                content=content,
                type=ftype,
                scope=self.scope,
                source="auto",
                confidence=confidence,
            )
            ids.append(self.store.save(entry))
        return ids

    # ------------------------------------------------------- loss audit

    def verify_retention(self, original_text: str, summary: str) -> float | None:
        """LLM-judge how much key info survived; None when no LLM / failure."""
        if self._llm is None or not original_text or not summary:
            return None
        try:
            resp = self._llm.chat(
                [
                    {"role": "system", "content": _RETENTION_PROMPT},
                    {
                        "role": "user",
                        "content": f"原文：\n{original_text[:4000]}\n\n摘要：\n{summary}",
                    },
                ],
                response_format={"type": "json_object"},
            )
            data = _extract_json(str(getattr(resp, "content", "")))
            return _clamp(data.get("retention", 0.0))
        except Exception:  # noqa: BLE001 - audit never blocks compression
            return None

    # ------------------------------------------------- context compress hook

    def on_compressed(self, old_messages: list[dict], summary: str) -> dict:
        """Called by ContextManager after a compression pass: demote the old
        turns to the memory DB and audit the retention. Best-effort."""
        try:
            ids = self.settle_facts(old_messages)
            retention = self.verify_retention(_flatten(old_messages), summary)
            if ids and retention is not None:
                # record the retention metric on the first saved fact
                first = self.store.get(ids[0])
                if first is not None:
                    self.store.update(
                        ids[0],
                        metadata={**first.metadata, "retention": retention},
                    )
            return {"facts_saved": len(ids), "retention": retention}
        except Exception:  # noqa: BLE001
            return {"facts_saved": 0, "retention": None}

    # --------------------------------------------- memory DB rolling update

    def summarize_cluster(
        self,
        *,
        scope: str | None = None,
        min_count: int = 5,
        max_entries: int = 20,
        max_confidence: float = 0.7,
    ) -> int:
        """Merge low-confidence auto memories of one scope into a single summary
        entry and DEMOTE the originals (deprecated_by='merged:<id>') instead of
        deleting them. Returns the number of memories merged (0 = nothing to
        do or no LLM)."""
        if self._llm is None:
            return 0
        scope = scope or self.scope
        active = self.store.list(scope=scope, include_deprecated=False)
        candidates = [
            e
            for e in active
            if e.source == "auto" and float(e.confidence or 0.0) < max_confidence
        ]
        if len(candidates) < min_count:
            return 0
        pool = candidates[:max_entries]
        text = "\n".join(f"- {e.content[:120]}" for e in pool)
        try:
            resp = self._llm.chat(
                [
                    {"role": "system", "content": _SUMMARY_PROMPT},
                    {"role": "user", "content": text[:6000]},
                ],
                response_format={"type": "json_object"},
            )
            data = _extract_json(str(getattr(resp, "content", "")))
            summary = str(data.get("summary", "")).strip()
            if not summary:
                return 0
            summary_id = self.store.save(
                MemoryEntry(
                    content=summary,
                    type="decision",
                    scope=scope,
                    source="auto",
                    confidence=0.5,
                    metadata={"merged": len(pool)},
                )
            )
            for e in pool:
                self.store.update(e.id, deprecated_by=f"merged:{summary_id}")
            return len(pool)
        except Exception:  # noqa: BLE001 - merging is best-effort
            return 0

    # ------------------------------------------------------------- rules

    @staticmethod
    def _rule_facts(messages: list[dict]) -> list[dict]:
        """Zero-LLM fallback: surface file paths, error lines and decisions."""
        import re as _re

        files_seen: set[str] = set()
        errors: list[str] = []
        for m in messages:
            text = m.get("content", "") or ""
            if not isinstance(text, str):
                continue
            for match in _re.finditer(r"[\w./\-]+\.\w{1,5}", text):
                files_seen.add(match.group())
            for line in text.splitlines():
                if "error" in line.lower():
                    errors.append(line.strip()[:150])
        facts = []
        for f in sorted(files_seen)[:8]:
            facts.append({"fact": f"涉及文件：{f}", "type": "project", "confidence": 0.5})
        for e in errors[:5]:
            facts.append({"fact": f"错误：{e}", "type": "pattern", "confidence": 0.5})
        return facts


def _as_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.5
    return max(lo, min(v, hi))
