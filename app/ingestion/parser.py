from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader

# ---------------------------------------------------------------------------
# Structural markers present in the Schweser Notes (verified against the PDFs)
# ---------------------------------------------------------------------------
RE_READING = re.compile(r"^READING\s+(\d+)\s*$")
RE_MODULE = re.compile(r"^MODULE\s+([0-9.]+)\s*:\s*(.+?)\s*$", re.IGNORECASE)
RE_LO = re.compile(r"^LO\s+(\d+\.[a-z])\b(?:\s*:\s*)?(.*)$", re.IGNORECASE)
RE_STUDY_SESSION = re.compile(r"^STUDY\s+SESSION\s+\d+\b.*$", re.IGNORECASE)

# Section tags (the part of a module/reading a chunk belongs to). These let a
# retrieval request narrow the search to e.g. only "key_concepts" or "formulas".
SECTION_EXAM_FOCUS = "exam_focus"
SECTION_MODULE_CONTENT = "module_content"
SECTION_KEY_CONCEPTS = "key_concepts"
SECTION_FORMULAS = "formulas"
SECTION_MODULE_QUIZ = "module_quiz"
SECTION_ANSWER_KEY = "answer_key"
SECTION_INDEX = "index"

_HEADING_SECTIONS = (
    ("EXAM FOCUS", SECTION_EXAM_FOCUS),
    ("KEY CONCEPTS", SECTION_KEY_CONCEPTS),
    ("FORMULAS", SECTION_FORMULAS),
    ("MODULE QUIZ", SECTION_MODULE_QUIZ),
    ("ANSWER KEY", SECTION_ANSWER_KEY),
    ("INDEX", SECTION_INDEX),
)


def _heading_section(line: str) -> str | None:
    upper = line.upper()
    for marker, section in _HEADING_SECTIONS:
        if upper.startswith(marker):
            return section
    return None


@dataclass
class Block:
    """One paragraph-sized unit carrying its full structural context."""

    text: str
    page_number: int
    reading_no: int | None = None
    reading_title: str | None = None
    module_no: str | None = None
    module_title: str | None = None
    lo_code: str | None = None
    section: str | None = None


@dataclass
class ParsedDocument:
    book_no: int
    book_title: str
    pages: int
    blocks: list[Block] = field(default_factory=list)


def _is_noise(line: str) -> bool:
    """Header/footer/copyright fragments that should never become content."""
    if line.isdigit():
        return True
    if "©" in line or "ISBN" in line or "Schweser.com" in line:
        return True
    if line.startswith("Book ") and re.match(r"^Book\s+\d", line):
        return True
    if re.match(r"^SchweserNotes", line):
        return True
    if "Kaplan, Inc." in line or "GARP" in line:
        return True
    return False


def _normalize_lines(lines: list[str]) -> list[str]:
    """Join wrapped lines into sentences-ish text lines (single spaces)."""
    return [re.sub(r"\s+", " ", ln).strip() for ln in lines if ln.strip()]


def _book_no_from_filename(path: Path) -> int:
    m = re.search(r"Book\s+(\d+)", path.name, re.IGNORECASE)
    return int(m.group(1)) if m else 0


def parse_pdf(path: Path) -> ParsedDocument:
    """Extract text from a PDF and build a structure-aware block tree.

    Uses layout-mode extraction (preserves blank lines between paragraphs) and
    regex markers for READING / MODULE / LO to attach structural metadata to
    every paragraph block. Chunk sizes will therefore follow the document's own
    structure (variable sizes), not a fixed token window.
    """
    reader = PdfReader(str(path))
    doc = ParsedDocument(
        book_no=_book_no_from_filename(path),
        book_title=path.stem,
        pages=len(reader.pages),
    )

    reading_no: int | None = None
    reading_title: str | None = None
    module_no: str | None = None
    module_title: str | None = None
    lo_code: str | None = None
    section: str | None = None
    expect_reading_title = False
    first_body_page: int | None = None
    para: list[str] = []
    para_page = 1

    def flush() -> None:
        nonlocal para
        text = " ".join(_normalize_lines(para)).strip()
        if text:
            doc.blocks.append(
                Block(
                    text=text,
                    page_number=para_page,
                    reading_no=reading_no,
                    reading_title=reading_title,
                    module_no=module_no,
                    module_title=module_title,
                    lo_code=lo_code,
                    section=section,
                )
            )
        para = []

    for page_no, page in enumerate(reader.pages, start=1):
        raw = page.extract_text(extraction_mode="layout") or ""
        # Every reading body page opens with this boilerplate; pages before the
        # first occurrence are front matter (welcome/copyright/TOC/LO lists).
        if first_body_page is None and "following is a review of the" in raw.lower():
            first_body_page = page_no
            flush()
            reading_no = None
            reading_title = None
            module_no = None
            module_title = None
            lo_code = None
            section = None
        for raw_line in raw.split("\n"):
            line = raw_line.strip()
            if not line:
                flush()
                continue
            if _is_noise(line):
                continue

            if expect_reading_title:
                if RE_MODULE.match(line) or RE_LO.match(line) or RE_STUDY_SESSION.match(line):
                    pass  # fall through to marker handling
                else:
                    reading_title = re.sub(r"\s+", " ", line)
                    expect_reading_title = False
                    continue

            m = RE_READING.match(line)
            if m:
                flush()
                reading_no = int(m.group(1))
                reading_title = None
                module_no = None
                module_title = None
                lo_code = None
                section = SECTION_MODULE_CONTENT
                expect_reading_title = True
                continue

            m = RE_MODULE.match(line)
            if m:
                flush()
                module_no = m.group(1)
                module_title = re.sub(r"\s+", " ", m.group(2))
                lo_code = None
                section = SECTION_MODULE_CONTENT
                continue

            m = RE_LO.match(line)
            if m:
                flush()
                lo_code = m.group(1).lower()
                rest = m.group(2).strip()
                if rest:
                    para.append(rest)
                    para_page = page_no
                continue

            sec = _heading_section(line)
            if sec is not None or RE_STUDY_SESSION.match(line):
                flush()
                if sec is not None:
                    section = sec
                continue

            if not para:
                para_page = page_no
            para.append(line)
        flush()

    # Drop front matter: keep only blocks on or after the first body page, and
    # drop the boilerplate sentence that opens each reading.
    if first_body_page is not None:
        doc.blocks = [
            b
            for b in doc.blocks
            if b.page_number >= first_body_page
            and not b.text.lower().startswith("the following is a review of the")
        ]
    else:
        doc.blocks = []

    return doc