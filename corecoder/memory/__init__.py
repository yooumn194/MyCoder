"""Phase 5: hybrid-retrieval cross-session memory (zero-infrastructure).

Modules:
  types        — MemoryEntry / PatternMemory data models
  tokenizer    — jieba -> bigram fallback tokenization for FTS5
  security     — sensitive-info redaction before persist
  embedder     — fastembed / sentence-transformers / hashing backends + LRU
  store        — dual-db (project + global) SQLite + FTS5 + vector backends
  retriever    — BM25 + vector, fused with RRF(k=60)
  maintenance  — confidence decay, compaction, stats
  prompt       — bounded memory section for the system prompt
  integration  — wiring into planning_guard / Self-Correction / plan.json
"""

from .retriever import HybridRetriever
from .store import MemoryStore, get_store, reset_store
from .types import MemoryEntry, PatternMemory

__all__ = [
    "MemoryStore",
    "MemoryEntry",
    "PatternMemory",
    "HybridRetriever",
    "get_store",
    "reset_store",
]
