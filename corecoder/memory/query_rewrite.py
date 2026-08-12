"""Query rewriting for multi-turn retrieval (P0, query_rewrite).

In a conversation the user's latest message is often not self-contained
("那它呢？" / "和 RAG 比呢？"). Retrieval quality collapses if we search on
the raw fragment. A QueryRewriter merges the recent history with the current
message into a standalone retrieval query.

Pluggable like retriever.Reranker: IdentityQueryRewriter is the default no-op;
LLMQueryRewriter uses one cheap LLM call (json_object) and falls back to the
original query on any failure, so retrieval never breaks because of rewriting.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod

from corecoder.config import Config
from corecoder.llm import LLM

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

_SYSTEM_PROMPT = """\
你是检索查询改写器。把「多轮对话历史 + 当前用户问题」改写成一个独立、自包含的检索查询，用于向量/关键词检索。

要求：
1. 严格输出 JSON：{"query": "改写后的查询"}
2. 把历史里的指代（它/这个/那/和XX比）替换成具体实体，补全省略的上下文
3. 不要编造历史里没有的信息；如果当前问题本身自包含，原样返回
4. 输出为中文检索词，长度不超过 50 字"""


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


def _format_history(history: list[dict] | None) -> str:
    if not history:
        return "(无)"
    parts = []
    for turn in history[-6:]:  # last few turns are enough to resolve references
        role = turn.get("role", "")
        content = str(turn.get("content", ""))[:300]
        if content:
            parts.append(f"{role}: {content}")
    return "\n".join(parts) if parts else "(无)"


class QueryRewriter(ABC):
    @abstractmethod
    def rewrite(self, query: str, history: list[dict] | None = None) -> str:
        """Return a standalone retrieval query."""


class IdentityQueryRewriter(QueryRewriter):
    """Default no-op — the query is used as-is."""

    def rewrite(self, query: str, history: list[dict] | None = None) -> str:
        return query


class LLMQueryRewriter(QueryRewriter):
    """Rewrite via one cheap LLM call; any failure returns the original query."""

    def __init__(self, llm: LLM | None = None, timeout_seconds: float = 15.0) -> None:
        self._llm = llm if llm is not None else _resolve_llm()
        self.timeout = timeout_seconds

    def rewrite(self, query: str, history: list[dict] | None = None) -> str:
        if self._llm is None:
            return query
        if not query or not query.strip():
            return query
        user = (
            f"对话历史：\n{_format_history(history)}\n\n"
            f"当前用户问题：{query}"
        )
        try:
            raw = self._llm.chat(
                [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
            )
            data = _extract_json(str(getattr(raw, "content", "")))
            rewritten = str(data.get("query") or "").strip()
            return rewritten if rewritten else query
        except Exception:  # noqa: BLE001 - rewriting is best-effort
            return query


def _resolve_llm() -> LLM | None:
    cfg = Config.from_env()
    if not cfg.api_key:
        return None
    return LLM(
        model=cfg.model,
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )
