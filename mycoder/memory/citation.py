"""Citation integrity for retrieval-grounded answers (P2).

Mirrors Claude Code's stance that a grounded answer must be traceable — search
with deterministic tools, then cite: the prompt forces [n] citations with an
explicit "no evidence" fallback, and verify_citations() post-hoc checks that
every cited id actually exists in the supplied contexts (a cited id outside
the context range is a hallucinated reference).

The verdict plugs straight into the eval bench as a scoring dimension: an
answer whose [n]s all resolve and whose claims carry citations scores higher
than one that cites nothing or cites phantom ids.
"""

from __future__ import annotations

import re

_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

GROUNDED_ANSWER_PROMPT = """\
基于以下检索结果回答问题。每个事实性陈述必须标注来源编号 [n]（n 为检索结果的下标，从 1 开始）。
如果检索结果不足以回答，明确说"未找到相关信息"，不要编造。

检索结果：
{contexts}"""


def extract_citation_ids(answer: str | None) -> list[int]:
    """All cited context indices, in order of appearance.

    "[1] 是 A，[2, 3] 是 B" -> [1, 2, 3].
    """
    ids: list[int] = []
    for m in _CITATION_RE.finditer(answer or ""):
        for part in m.group(1).split(","):
            try:
                ids.append(int(part.strip()))
            except ValueError:  # pragma: no cover - regex already guarantees digits
                continue
    return ids


def verify_citations(answer: str | None, n_contexts: int) -> dict:
    """Validate an answer's citations against the number of provided contexts.

    Returns {valid, cited, context_count, missing, citation_density}. A citation
    index outside [1, n_contexts] is a hallucinated reference (missing); a valid
    answer must have zero missing AND at least one citation.
    """
    cited = extract_citation_ids(answer)
    missing = [c for c in cited if c < 1 or c > n_contexts]
    token_count = max(1, len((answer or "").split()))
    return {
        "valid": not missing and len(cited) > 0,
        "cited": cited,
        "context_count": n_contexts,
        "missing": missing,
        "citation_density": round(len(cited) / token_count, 3),
    }


def grounded_answer_prompt(contexts: list[str]) -> str:
    """Build the grounded-answer prompt with numbered contexts."""
    numbered = "\n".join(f"[{i + 1}] {str(c)[:200]}" for i, c in enumerate(contexts))
    return GROUNDED_ANSWER_PROMPT.format(contexts=numbered)
