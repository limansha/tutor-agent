"""Retrieval pipeline: embed query -> cosine search -> parent expansion."""

from __future__ import annotations

from app import db
from app.embeddings import embed_query
from app.retrieval import vectorstore
from app.schemas import ChatFilters, ChunkMetadata, SourceChunk


def _row_to_source(row: dict) -> SourceChunk:
    meta = ChunkMetadata(
        book_no=row.get("book_no"),
        book_title=row.get("book_title"),
        source_path=row.get("source_path"),
        year=row.get("year"),
        reading_no=row.get("reading_no"),
        reading_title=row.get("reading_title"),
        module_no=row.get("module_no"),
        module_title=row.get("module_title"),
        lo_code=row.get("lo_code"),
        section=row.get("section"),
        page_number=row.get("page_number"),
        chunk_index=row.get("chunk_index"),
        token_count=row.get("token_count"),
        doc_url=row.get("doc_url"),
        parent_chunk_id=row.get("parent_id"),
    )
    return SourceChunk(
        chunk_id=row.get("id"),
        role=row.get("role", "child"),
        text=row.get("text", ""),
        score=row.get("score"),
        metadata=meta,
    )


def retrieve(
    query: str,
    filters: ChatFilters | None = None,
    top_k: int = 8,
    parent_expansion: bool = True,
    section_expansion: bool = False,
) -> list[SourceChunk]:
    """Embed the query, cosine-search child chunks, expand to parent contexts."""
    if not db.is_ready():
        raise db.DatabaseNotReady(db.readiness_hint())

    query_embedding = embed_query(query)

    rows = vectorstore.similarity_search(
        query_embedding,
        top_k=top_k,
        filters=filters.model_dump(exclude_none=True) if filters else None,
        parent_expansion=parent_expansion,
        section_expansion=section_expansion,
    )
    return [_row_to_source(r) for r in rows]