from __future__ import annotations

import re
from dataclasses import dataclass

from app.config import config
from app.ingestion.parser import Block, ParsedDocument
from app.tokenize import count_tokens

_ABBREVIATIONS = re.compile(
    r"\b(?:Mr|Ms|Mrs|Dr|St|Jr|Sr|vs|etc|No|Inc|Ltd|i\.e|e\.g)\.", re.IGNORECASE
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def _split_sentences(text: str) -> list[str]:
    """Sentence splitter protecting common abbreviations (no external deps)."""
    protected = _ABBREVIATIONS.sub(lambda m: m.group(0).replace(".", "<DOT>"), text)
    parts = _SENTENCE_SPLIT.split(protected)
    return [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]


@dataclass
class ChunkRecord:
    """A chunk ready to be embedded and stored (parent or child)."""

    role: str  # 'parent' | 'child'
    text: str
    page_number: int
    chunk_index: int
    token_count: int
    parent_ref: int | None = None  # 0-based index into parent list (children only)
    reading_no: int | None = None
    reading_title: str | None = None
    module_no: str | None = None
    module_title: str | None = None
    lo_code: str | None = None
    section: str | None = None


@dataclass
class _Sentence:
    """One sentence inside a parent, carrying its block-level metadata."""

    text: str
    page_number: int
    lo_code: str | None = None
    section: str | None = None


@dataclass
class _Unit:
    """A parent-sized piece of a section: blocks (paragraph-preserving) or sentences."""

    items: list  # list[Block] | list[_Sentence]
    reading_no: int | None = None
    reading_title: str | None = None
    module_no: str | None = None
    module_title: str | None = None


def _section_key(block: Block) -> tuple[int | None, str | None]:
    """Consecutive blocks sharing reading+module belong to the same section."""
    return block.reading_no, block.module_no


def _parent_blocks(doc: ParsedDocument) -> list[list[Block]]:
    groups: list[list[Block]] = []
    for block in doc.blocks:
        if not groups or _section_key(groups[-1][0]) != _section_key(block):
            groups.append([block])
        else:
            groups[-1].append(block)
    return [g for g in groups if g]


def _unit_tokens(unit: _Unit) -> int:
    if isinstance(unit.items[0], Block):
        return sum(count_tokens(b.text) for b in unit.items)
    return sum(count_tokens(s.text) for s in unit.items)


def _unit_text(unit: _Unit) -> str:
    if isinstance(unit.items[0], Block):
        return "\n\n".join(b.text for b in unit.items).strip()
    return " ".join(s.text for s in unit.items).strip()


def _unit_meta(unit: _Unit) -> tuple:
    head = unit.items[0]
    return (
        head.page_number,
        unit.reading_no,
        unit.reading_title,
        unit.module_no,
        unit.module_title,
        head.lo_code,
        head.section,
    )


def _pack_sentences(sentences: list[_Sentence], max_tokens: int) -> list[list[_Sentence]]:
    """Greedily pack sentences into <= max_tokens parts at sentence boundaries."""
    parts: list[list[_Sentence]] = []
    current: list[_Sentence] = []
    current_tokens = 0
    for sentence in sentences:
        tokens = count_tokens(sentence.text)
        if current and current_tokens + tokens > max_tokens:
            parts.append(current)
            current, current_tokens = [], 0
        current.append(sentence)
        current_tokens += tokens
    if current:
        parts.append(current)
    return parts


def _split_section_units(section: list[Block], max_tokens: int) -> list[_Unit]:
    """A section becomes one unit, unless oversized -> sentence-level parts.

    Splits oversized sections at full-stop (sentence) boundaries so every
    parent lands in [MIN_CHUNK_TOKENS, PARENT_MAX_TOKENS].
    """
    first = section[0]
    meta = dict(
        reading_no=first.reading_no,
        reading_title=first.reading_title,
        module_no=first.module_no,
        module_title=first.module_title,
    )
    total = sum(count_tokens(b.text) for b in section)
    if total <= max_tokens:
        return [_Unit(section, **meta)]
    sentences = [
        _Sentence(s, b.page_number, b.lo_code, b.section)
        for b in section
        for s in _split_sentences(b.text)
    ]
    return [_Unit(part, **meta) for part in _pack_sentences(sentences, max_tokens)]


def _merge_small_units(units: list[_Unit], min_tokens: int) -> list[_Unit]:
    """Absorb sub-min units into the previous unit (keeps section ordering)."""
    merged: list[_Unit] = []
    for unit in units:
        if _unit_tokens(unit) >= min_tokens or not merged:
            merged.append(unit)
            continue
        prev = merged[-1]
        merged[-1] = _Unit(
            [*prev.items, *unit.items],
            reading_no=prev.reading_no,
            reading_title=prev.reading_title,
            module_no=prev.module_no,
            module_title=prev.module_title,
        )
    return merged


def _merge_tiny_sections(sections: list[list[Block]], min_tokens: int) -> list[list[Block]]:
    """Combine genuinely tiny sections (<min tokens) into the next section of the
    same reading (or backward into the previous one if none follows)."""
    merged: list[list[Block]] = []
    i = 0
    n = len(sections)
    while i < n:
        sec = sections[i]
        tokens = sum(count_tokens(b.text) for b in sec)
        if tokens >= min_tokens:
            merged.append(sec)
            i += 1
            continue
        target = None
        for j in range(i + 1, n):
            if sections[j][0].reading_no == sec[0].reading_no:
                target = j
                break
        if target is not None:
            sections[target] = [*sec, *sections[target]]
        elif merged:
            merged[-1] = [*merged[-1], *sec]
        else:
            merged.append(sec)
        i += 1
    return merged


def _sentences_of(unit: _Unit) -> list[_Sentence]:
    if isinstance(unit.items[0], Block):
        return [
            _Sentence(s, b.page_number, b.lo_code, b.section)
            for b in unit.items
            for s in _split_sentences(b.text)
        ]
    return unit.items


def _child_windows(sentences: list[_Sentence]) -> list[tuple[str, int, str | None, str | None]]:
    """Variable-size sliding windows merged at the source.

    Each window starts at a slide offset and is *extended* (never concatenated
    with another window) until it reaches MIN_CHUNK_TOKENS or the parent ends,
    then capped at CHILD_MAX_TOKENS. A too-small final window is absorbed by the
    previous one (extending its end) so no child falls below MIN_CHUNK_TOKENS
    and no text is duplicated.
    """
    ranges: list[list[int]] = []
    i = 0
    n = len(sentences)

    def tokens(lo: int, hi: int) -> int:
        return count_tokens(" ".join(s.text for s in sentences[lo:hi]))

    while i < n:
        end = min(i + config.CHILD_WINDOW_SENTENCES, n)
        while tokens(i, end) < config.MIN_CHUNK_TOKENS and end < n:
            end += 1
        while end - i > 1 and tokens(i, end) > config.CHILD_MAX_TOKENS:
            end -= 1
        if end > i:
            ranges.append([i, end])
        if end >= n:
            break
        i += config.CHILD_SLIDE_SENTENCES

    if len(ranges) > 1 and tokens(*ranges[-1]) < config.MIN_CHUNK_TOKENS:
        ranges[-2][1] = ranges[-1][1]
        ranges.pop()

    out: list[tuple[str, int, str | None, str | None]] = []
    for start, end in ranges:
        window = sentences[start:end]
        joined = " ".join(s.text for s in window).strip()
        if not joined:
            continue
        head = window[0]
        out.append((joined, head.page_number, head.lo_code, head.section))
    return out


def chunk_document(doc: ParsedDocument) -> tuple[list[ChunkRecord], list[ChunkRecord]]:
    """Structure-aware parent-child + merge-at-source sliding-window chunking.

    - Parent  = a section (module / reading block). Sections below
      MIN_CHUNK_TOKENS are combined into the next same-reading section;
      oversized sections (> PARENT_MAX_TOKENS) are split in half at sentence
      boundaries. Resulting parent sizes stay in [MIN_CHUNK_TOKENS,
      PARENT_MAX_TOKENS].
    - Child   = a window of sentences inside one parent, started every
      CHILD_SLIDE_SENTENCES and extended at the source until it reaches
      MIN_CHUNK_TOKENS (capped at CHILD_MAX_TOKENS). Windows never cross
      section boundaries.
    """
    parents: list[ChunkRecord] = []
    children: list[ChunkRecord] = []

    for section in _merge_tiny_sections(_parent_blocks(doc), config.MIN_CHUNK_TOKENS):
        for unit in _merge_small_units(
            _split_section_units(section, config.PARENT_MAX_TOKENS),
            config.MIN_CHUNK_TOKENS,
        ):
            parent_text = _unit_text(unit)
            if not parent_text:
                continue
            page, rno, rtitle, mno, mtitle, lo, sec = _unit_meta(unit)
            parent = ChunkRecord(
                role="parent",
                text=parent_text,
                page_number=page,
                chunk_index=len(parents),
                token_count=count_tokens(parent_text),
                reading_no=rno,
                reading_title=rtitle,
                module_no=mno,
                module_title=mtitle,
                lo_code=lo,
                section=sec,
            )
            parents.append(parent)

            for window_text, wpage, wlo, wsec in _child_windows(_sentences_of(unit)):
                children.append(
                    ChunkRecord(
                        role="child",
                        text=window_text,
                        page_number=wpage,
                        chunk_index=len(children),
                        token_count=count_tokens(window_text),
                        parent_ref=len(parents) - 1,
                        reading_no=rno,
                        reading_title=rtitle,
                        module_no=mno,
                        module_title=mtitle,
                        lo_code=wlo,
                        section=wsec,
                    )
                )

    return parents, children