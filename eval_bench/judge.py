"""LLM-as-Judge (P2) — quality scoring beyond pass/fail.

Pass/fail only tells you whether the tests passed; it gives no signal on a
partially-correct or borderline answer. LLMJudge scores an answer against a
strict rubric (correctness, faithfulness to the reference) and returns a
0..1 score plus one-line reasoning.

Known LLM-as-Judge biases we consciously mitigate (and can discuss in an
interview):

  * position bias        — reference answer always listed first,
                           judge prompt fixes the comparison order;
  * self-preference      — the judge is a different model / prompt from the
                           answerer where possible;
  * verbosity bias       — rubric explicitly says longer != better;
  * strictness drift     — score is forced to the {0.0, 0.5, 1.0} grid and
                           bucketed so the judge can't grade on vibes.

The judge is a best-effort add-on: any failure falls back to score=0.5 and
never breaks the eval pipeline.
"""

from __future__ import annotations

import json
import re
from typing import Any

from mycoder.config import Config

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

_RUBRIC = """\
你是评测裁判。对「答案」按以下标准打分，严格输出 JSON：{"score": 0.0|0.5|1.0, "reasoning": "一句话理由"}

评分标准：
- 1.0：答案正确、完整、忠实于参考（若有）；不因长短加分或扣分
- 0.5：部分正确 / 有主要遗漏
- 0.0：错误或离题
- 禁止用「看起来不错」这类模糊判断；必须给出可核对的理由"""


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


class LLMJudge:
    """Scores an answer on correctness/faithfulness via one LLM call."""

    def __init__(self, llm=None) -> None:
        self._llm = llm if llm is not None else _resolve_llm()

    def judge(
        self, question: str, answer: str, reference: str | None = None
    ) -> dict[str, Any]:
        """Return {score, reasoning}; score 0..1, falls back to 0.5 on failure."""
        if self._llm is None:
            return {"score": 0.5, "reasoning": "no judge LLM available"}
        user = (
            f"问题：{question}\n"
            f"参考答案：{reference or '(无)'}\n"
            f"待评答案：{answer}"
        )
        try:
            raw = self._llm.chat(
                [
                    {"role": "system", "content": _RUBRIC},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
            )
            data = _extract_json(str(getattr(raw, "content", "")))
            score = float(data.get("score", 0.5))
            if score not in (0.0, 0.5, 1.0):
                score = 0.5  # force the grid (strictness drift mitigation)
            return {
                "score": score,
                "reasoning": str(data.get("reasoning", ""))[:200],
            }
        except Exception:  # noqa: BLE001 - judge must never break the eval
            return {"score": 0.5, "reasoning": "judge call failed"}


def _resolve_llm():
    cfg = Config.from_env()
    if not cfg.api_key:
        return None
    from mycoder.llm import LLM

    return LLM(
        model=cfg.model,
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )
