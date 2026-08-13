"""Document-level RAG layer on top of the hybrid memory store.

Chunks a document (Markdown headings, top-level Python def/class, code fences,
paragraphs) into overlapping pieces; stores each chunk as an ordinary memory
entry (type='fact', deterministic id `doc:{doc_id}:{index}`, metadata carries
doc_id / chunk_index / chunk_hash); and supports incremental reindexing so only
changed chunks are re-vectorized / re-indexed.

Because chunks are normal memories, they flow through the existing
HybridRetriever — the RuleReranker (retriever.py) re-orders them by query-term
hits + length penalty after RRF fusion.

Design notes:
  * Deterministic chunk ids make upserts idempotent — save_document / reindex
    never duplicate.
  * `store.save(dedup=False)` bypasses cosine dedup for document chunks: two
    chunks legitimately share wording (e.g. overlap), and content-similarity
    dedup would wrongly merge them and break the id scheme.
  * Chunk hash (sha256) in metadata lets reindex skip untouched chunks — only
    changed ones go through store.update (which rewrites FTS + embedding).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .types import MemoryEntry

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .store import MemoryStore

DEFAULT_MAX_CHARS = 1500
DEFAULT_OVERLAP_RATIO = 0.15

# A line that starts a semantic block.
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+\S")
_TOP_DEF = re.compile(r"^(?:def|class)\s+\w+")
_FENCE = re.compile(r"^\s*(```|~~~)")


@dataclass
class Chunk:
    """One document chunk (a memory entry payload)."""

    index: int
    text: str
    start_line: int
    end_line: int
    heading: str | None = None
    # Parent-child index (P2): the chunk index where this chunk's section
    # starts. The section's first chunk is the "parent" (larger context);
    # later chunks are "children" that reference it via metadata.parent_id.
    section_start: int = 0


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def chunk_id(doc_id: str, index: int) -> str:
    return f"doc:{doc_id}:{index}"


def _split_blocks(text: str) -> list[tuple[int, str]]:
    """Split text into semantic blocks: (start_line, text).

    A new block starts at a Markdown heading, a top-level `def`/`class`, or
    after a blank line (paragraph break). Fenced code blocks (```) are treated
    as single units and never split internally.
    """
    lines = text.split("\n")
    blocks: list[tuple[int, str]] = []
    start = 0
    current: list[str] = []
    in_fence = False
    fence_marker: str | None = None

    def flush() -> None:
        nonlocal start, current
        if current:
            blocks.append((start, "\n".join(current).rstrip("\n")))
        start = 0
        current = []

    for i, line in enumerate(lines):
        match = _FENCE.match(line)
        if match:
            if not in_fence:
                in_fence = True
                fence_marker = match.group(1)
                flush()  # fence opener is a hard boundary
                start = i
            else:
                if match.group(1) == fence_marker:
                    in_fence = False
            current.append(line)
            continue

        is_boundary = not in_fence and (
            _HEADING.match(line)
            or _TOP_DEF.match(line)
            or (i > 0 and lines[i - 1].strip() == "")
        )
        if is_boundary and current:
            flush()
            start = i
        if not current:
            start = i
        current.append(line)

    flush()
    return blocks


def _first_heading(text: str) -> str | None:
    for line in text.splitlines():
        if _HEADING.match(line):
            return line.lstrip("# ").strip() or None
    return None


def _wrap_line(line: str, max_chars: int) -> list[str]:
    """Split one line that exceeds max_chars into ≤max_chars pieces.

    Word-wraps on spaces first, then char-slices any single word that is itself
    too long — so a giant unbroken paragraph is still chunked.
    """
    if len(line) <= max_chars:
        return [line]
    pieces: list[str] = []
    cur = ""
    for word in line.split(" "):
        if not cur:
            cur = word
        elif len(cur) + 1 + len(word) <= max_chars:
            cur += " " + word
        else:
            pieces.append(cur)
            cur = word
    if cur:
        pieces.append(cur)
    final: list[str] = []
    for piece in pieces:
        if len(piece) <= max_chars:
            final.append(piece)
        else:
            final.extend(piece[i : i + max_chars] for i in range(0, len(piece), max_chars))
    return final


def chunk_document(
    text: str,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
) -> list[Chunk]:
    """Split a document into overlapping chunks.

    Semantic blocks are greedily packed up to `max_chars`; when a block alone
    exceeds the budget it is split on line boundaries. `overlap_ratio` of the
    budget is carried into the following chunk (as whole lines) so context is
    not lost across a cut.
    """
    if not text or not text.strip():
        return []
    blocks = _split_blocks(text)

    # split oversized blocks into ≤max_chars units, keeping line integrity
    units: list[str] = []
    for _start, block in blocks:
        if len(block) <= max_chars:
            units.append(block)
            continue
        pieces: list[str] = []
        for line in block.split("\n"):
            pieces.extend(_wrap_line(line, max_chars))
        cur = ""
        for piece in pieces:
            if cur and len(cur) + 1 + len(piece) > max_chars:
                units.append(cur)
                cur = piece
            else:
                cur = f"{cur}\n{piece}" if cur else piece
        if cur:
            units.append(cur)

    units = [u for u in units if u.strip()]

    # greedy pack with a character-bounded overlap carried into the next chunk
    overlap_chars = max(0, int(max_chars * overlap_ratio))

    def _flush(buf_start: int, next_line: int, buf: list[str], chunks: list[Chunk]) -> tuple[int, list[str]]:
        text = "\n".join(buf)
        chunks.append(
            Chunk(
                index=len(chunks),
                text=text,
                start_line=buf_start,
                end_line=next_line - 1,
                heading=_first_heading(text),
            )
        )
        if not overlap_chars:
            return next_line, []
        tail = text[-overlap_chars:]
        nl = tail.find("\n")
        if nl >= 0:
            tail = tail[nl + 1:]  # re-align to a line boundary
        tail_lines = tail.split("\n") if tail else []
        return next_line - len(tail_lines), tail_lines

    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_start = 1
    next_line = 1
    for unit in units:
        unit_lines = unit.split("\n")
        if buf and len("\n".join(buf)) + 1 + len(unit) > max_chars:
            buf_start, buf = _flush(buf_start, next_line, buf, chunks)
        buf.extend(unit_lines)
        next_line += len(unit_lines)
    if buf:
        _flush(buf_start, next_line, buf, chunks)

    # assign section roots (parent-child index): a chunk that carries a heading
    # starts a new section; its children share the same section_start.
    section_start = 0
    for i, chunk in enumerate(chunks):
        if chunk.heading and i != 0:
            section_start = i
        chunk.section_start = section_start
    return chunks


def _entry_for(chunk: Chunk, doc_id: str, scope: str) -> MemoryEntry:
    parent_id = None
    if chunk.index != chunk.section_start:
        parent_id = chunk_id(doc_id, chunk.section_start)  # child -> section root
    return MemoryEntry(
        id=chunk_id(doc_id, chunk.index),
        content=chunk.text,
        type="fact",
        scope=scope,
        metadata={
            "doc_id": doc_id,
            "chunk_index": chunk.index,
            "chunk_hash": _hash(chunk.text),
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "parent_id": parent_id,
        },
    )


def retrieve_parents(store: "MemoryStore", results: list[dict]) -> list[dict]:
    """Parent-child RAG (P2): for each matched child chunk, attach its parent's
    (section-level, larger-context) content. A parent chunk is returned as-is
    with parent_content == its own text. Never raises on missing parents."""
    out: list[dict] = []
    for result in results:
        meta = result.get("metadata") or {}
        parent_id = meta.get("parent_id")
        parent_content = None
        if parent_id:
            parent = store.get(parent_id)
            if parent is not None:
                parent_content = parent.content
        out.append({**result, "parent_id": parent_id, "parent_content": parent_content})
    return out


def save_document(
    store: "MemoryStore",
    doc_id: str,
    text: str,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
    scope: str = "project",
) -> int:
    """Chunk a document and upsert every chunk as a memory. Returns the number
    of chunks written (existing chunk ids are updated, new ones inserted)."""
    chunks = chunk_document(text, max_chars=max_chars, overlap_ratio=overlap_ratio)
    for chunk in chunks:
        mem_id = chunk_id(doc_id, chunk.index)
        entry = _entry_for(chunk, doc_id, scope)
        if store.get(mem_id) is None:
            store.save(entry, dedup=False)
        else:
            store.update(mem_id, content=entry.content, metadata=entry.metadata)
    return len(chunks)


def reindex_document(
    store: "MemoryStore",
    doc_id: str,
    text: str,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
    scope: str = "project",
) -> dict[str, int]:
    """Incremental reindex: only chunks whose content hash changed are
    re-vectorized / re-indexed (via store.update). Returns a diff summary.

      {"added": n, "updated": n, "removed": n, "unchanged": n}
    """
    chunks = chunk_document(text, max_chars=max_chars, overlap_ratio=overlap_ratio)
    expected = {chunk_id(doc_id, c.index): _hash(c.text) for c in chunks}
    existing = {
        e.id: (e.metadata.get("chunk_hash"), e.metadata.get("chunk_index"))
        for e in store.list_by_metadata("doc_id", doc_id)
    }

    added = updated = removed = unchanged = 0
    for mem_id, content_hash in expected.items():
        prior = existing.get(mem_id)
        if prior is None:
            chunk = chunks[int(mem_id.rsplit(":", 1)[1])]
            store.save(_entry_for(chunk, doc_id, scope), dedup=False)
            added += 1
        elif prior[0] == content_hash:
            unchanged += 1
        else:
            chunk = chunks[int(mem_id.rsplit(":", 1)[1])]
            store.update(mem_id, content=chunk.text, metadata=_entry_for(chunk, doc_id, scope).metadata)
            updated += 1

    for mem_id in existing:
        if mem_id not in expected:
            store.delete(mem_id)
            removed += 1

    return {"added": added, "updated": updated, "removed": removed, "unchanged": unchanged}
