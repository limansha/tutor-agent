"""Ingestion pipeline: parse -> chunk -> embed -> store."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import quote

from app import db
from app.config import config
from app.embeddings import embed_documents
from app.ingestion.chunker import chunk_document
from app.ingestion.parser import parse_pdf
from app.retrieval import vectorstore
from app.schemas import FileIngestResult


def _doc_meta(doc, path: Path) -> dict:
    return {
        "book_no": doc.book_no,
        "book_title": doc.book_title,
        "source_path": str(path),
        "year": config.YEAR,
        "doc_prefix": f"/docs/{quote(path.name)}#page=",
    }


def _existing_sources() -> set[str]:
    if not db.is_ready():
        return set()
    conn = db.connect()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT source_path FROM frm_books")
            return {r[0] for r in cur.fetchall()}
    finally:
        conn.close()


def ingest_book(
    path: Path,
    force_reingest: bool,
    existing: set[str],
) -> FileIngestResult:
    result = FileIngestResult(filename=path.name, book_no=0)
    if not force_reingest and str(path) in existing:
        result.status = "skipped"
        result.detail = "already ingested (use force_reingest to rebuild)"
        return result

    doc = parse_pdf(path)
    result.book_no = doc.book_no
    result.pages = doc.pages

    parents, children = chunk_document(doc)
    records = parents + children
    if not records:
        result.status = "error"
        result.detail = "no text extracted"
        return result

    embeddings = embed_documents([r.text for r in records if r.role == "child"])
    if len(embeddings) != len([r for r in records if r.role == "child"]):
        result.status = "error"
        result.detail = "embedding count mismatch"
        return result

    emb_iter = iter(embeddings)
    record_embeddings = [next(emb_iter) if r.role == "child" else None for r in records]

    vectorstore.insert_chunks(records, record_embeddings, _doc_meta(doc, path))

    result.parents = len(parents)
    result.children = len(children)
    result.tokens = sum(r.token_count for r in records)
    result.status = "ok"
    return result


def ingest_books(
    books_dir: str,
    force_reingest: bool = False,
) -> list[FileIngestResult]:
    if not db.is_ready():
        raise db.DatabaseNotReady(db.readiness_hint())

    if force_reingest:
        vectorstore.delete_all()

    existing = set() if force_reingest else _existing_sources()
    pdf_files = sorted(
        p for p in Path(books_dir).glob("*.pdf") if re.search(r"Book\s+\d", p.name)
    )
    if not pdf_files:
        pdf_files = sorted(Path(books_dir).glob("*.pdf"))

    return [ingest_book(p, force_reingest, existing) for p in pdf_files]