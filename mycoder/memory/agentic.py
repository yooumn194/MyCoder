"""Agentic RAG (P1) — retrieve, judge sufficiency, re-query or stop.

Plain retrieval is single-shot: it returns whatever the query matched and never
asks "is this enough to answer?". AgenticRetriever wraps a HybridRetriever with
a loop:

    round N:  search(current_query)
              if sufficiency evaluator says "enough"  -> return
              else refine the query (broaden / rewrite) and try again
              until max_rounds

All components are pluggable so the loop is testable with zero LLM:
  * SufficiencyEvaluator  — RuleSufficiencyEvaluator (evidence count + top
                            confidence) is the default; an LLM judge can swap in.
  * QueryRefiner          — BroadeningRefiner (drop the most specific term to
                            widen recall) is the zero-dep default; LLMRefiner
                            reuses memory/query_rewrite for a semantic rewrite.

This answers the interview question "Agent 怎么判断检索结果够不够用、什么时候
该停止检索直接回答？" with an actual loop instead of a hand-wave.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .query_rewrite import LLMQueryRewriter, QueryRewriter
from .retriever import HybridRetriever
from .tokenizer import tokenize_chinese

DEFAULT_MAX_ROUNDS = 3


class SufficiencyEvaluator(ABC):
    """Decides whether the current evidence is enough to answer."""

    @abstractmethod
    def is_sufficient(
        self, query: str, results: list[dict], round_number: int
    ) -> bool:
        ...


class RuleSufficiencyEvaluator(SufficiencyEvaluator):
    """Heuristic: enough evidence when we have >= min_results results and the
    top result's confidence clears min_conf."""

    def __init__(self, min_results: int = 1, min_conf: float = 0.3) -> None:
        self.min_results = min_results
        self.min_conf = min_conf

    def is_sufficient(
        self, query: str, results: list[dict], round_number: int
    ) -> bool:
        if len(results) < self.min_results:
            return False
        top = max(float(r.get("confidence", 0.0)) for r in results)
        return top >= self.min_conf


class QueryRefiner(ABC):
    @abstractmethod
    def refine(self, query: str, results: list[dict], round_number: int) -> str:
        """Return the query for the next round."""


class BroadeningRefiner(QueryRefiner):
    """Zero-dep refiner: drop the most specific (last) query term to widen
    recall when the current query found nothing / too little."""

    def refine(self, query: str, results: list[dict], round_number: int) -> str:
        terms = tokenize_chinese(query).split()
        if len(terms) <= 1:
            return query
        return " ".join(terms[:-1])


class LLMRefiner(QueryRefiner):
    """Semantic refiner — delegates to a QueryRewriter (memory/query_rewrite)."""

    def __init__(self, rewriter: QueryRewriter | None = None) -> None:
        self.rewriter = rewriter if rewriter is not None else LLMQueryRewriter()

    def refine(self, query: str, results: list[dict], round_number: int) -> str:
        return self.rewriter.rewrite(query, history=None)


class AgenticRetriever:
    """Retrieve -> judge sufficiency -> refine or stop, up to max_rounds."""

    def __init__(
        self,
        retriever: HybridRetriever,
        *,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
        evaluator: SufficiencyEvaluator | None = None,
        refiner: QueryRefiner | None = None,
        min_conf: float = 0.1,
    ) -> None:
        self.retriever = retriever
        self.max_rounds = max_rounds
        self.evaluator = (
            evaluator if evaluator is not None else RuleSufficiencyEvaluator()
        )
        self.refiner = refiner if refiner is not None else BroadeningRefiner()
        self.min_conf = min_conf

    def search(
        self,
        query: str,
        limit: int = 10,
        scope: str | None = None,
        types: set[str] | list[str] | None = None,
    ) -> list[dict]:
        """Agentic search. Returns the evidence when judged sufficient, or the
        best-effort result after max_rounds of refinement."""
        if not query or not query.strip():
            return []
        current = query
        for round_number in range(1, self.max_rounds + 1):
            results = self.retriever.search(
                current, limit=limit, scope=scope, types=types, min_conf=self.min_conf
            )
            if self.evaluator.is_sufficient(current, results, round_number):
                return results
            if round_number >= self.max_rounds:
                return results
            current = self.refiner.refine(current, results, round_number)
        return []
