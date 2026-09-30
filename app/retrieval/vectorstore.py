"""Vector store operations (pgvector cosine similarity) + metadata filtering.

Real implementations, guarded by db.is_ready(). Invoked by the ingestion and
chat pipelines only after `make db-setup` has been run.
"""

from __future__ import annotations

from typing import Any

from app import db
from app.ingestion.chunker import ChunkRecord

_CHUNK_COLS = (
    "id",
    "role",
    "parent_id",
    "book_no",
    "book_title",
    "source_path",
    "year",
    "reading_no",
    "reading_title",
    "module_no",
    "module_title",
    "lo_code",
    "section",
    "page_number",
    "chunk_index",
    "token_count",
    "doc_url",
    "text",
)


def insert_chunks(
    records: list[ChunkRecord],
    embeddings: list[list[float] | None],
    doc_meta: dict[str, Any],
) -> None:
    """Insert parent chunks first (RETURNING id), then children linked by parent_id.

    Parents carry no embedding (embedding=None -> NULL) and are excluded from
    the HNSW index (partial index WHERE role='child').
    """
    if not records:
        return
    conn = db.connect()
    try:
        with conn.cursor() as cur:
            parents: list[tuple] = []
            children: list[tuple] = []
            for rec, emb in zip(records, embeddings):
                emb_sql = None if emb is None else f"[{','.join(f'{x:.6f}' for x in emb)}]"
                row = (
                    rec.role,
                    rec.page_number,
                    rec.chunk_index,
                    rec.token_count,
                    rec.reading_no,
                    rec.reading_title,
                    rec.module_no,
                    rec.module_title,
                    rec.lo_code,
                    rec.section,
                    doc_meta["book_no"],
                    doc_meta["book_title"],
                    doc_meta["source_path"],
                    doc_meta["year"],
                    f"{doc_meta['doc_prefix']}{rec.page_number}",
                    rec.text,
                    emb_sql,
                )
                (parents if rec.role == "parent" else children).append(row)

            parent_ids: dict[int, int] = {}  # parent_ref -> db id
            for row in parents:
                cur.execute(
                    """
                    INSERT INTO frm_books
                        (role, page_number, chunk_index, token_count,
                         reading_no, reading_title, module_no, module_title, lo_code, section,
                         book_no, book_title, source_path, year, doc_url, text, embedding)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::vector)
                    RETURNING id
                    """,
                    row,
                )
                db_id = cur.fetchone()[0]
                parent_ids[len(parent_ids)] = db_id

            child_records = [r for r in records if r.role == "child"]
            for rec, row in zip(child_records, children):
                parent_id = parent_ids.get(rec.parent_ref) if rec.parent_ref is not None else None
                cur.execute(
                    """
                    INSERT INTO frm_books
                        (role, parent_id, page_number, chunk_index, token_count,
                         reading_no, reading_title, module_no, module_title, lo_code, section,
                         book_no, book_title, source_path, year, doc_url, text, embedding)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::vector)
                    """,
                    (rec.role, parent_id, *row[1:]),
                )
        conn.commit()
    finally:
        conn.close()


def similarity_search(
    query_embedding: list[float],
    top_k: int = 8,
    filters: dict[str, Any] | None = None,
    parent_expansion: bool = True,
    section_expansion: bool = False,
) -> list[dict]:
    """Cosine similarity over child chunks, then fetch linked parent contexts.

    With `section_expansion`, also returns every sibling chunk (both roles) that
    shares the (book_no, reading_no, module_no) of a matched child — the complete
    content of the module the match landed in.
    """
    conn = db.connect()
    try:
        emb_sql = f"[{','.join(f'{x:.6f}' for x in query_embedding)}]"
        where, params = _filter_clause(filters or {})

        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT {', '.join(_CHUNK_COLS)},
                       1 - (embedding <=> %s::vector) AS score
                FROM frm_books
                WHERE role = 'child'
                  {where}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (emb_sql, *params, emb_sql, top_k),
            )
            children = [
                {**dict(zip(_CHUNK_COLS + ("score",), row))}
                for row in cur.fetchall()
            ]

            if parent_expansion and children:
                parent_ids = {c["parent_id"] for c in children if c["parent_id"]}
                parents: list[dict] = []
                if parent_ids:
                    cur.execute(
                        f"""
                        SELECT {', '.join(_CHUNK_COLS)}
                        FROM frm_books
                        WHERE id = ANY(%s)
                        """,
                        (list(parent_ids),),
                    )
                    parents = [
                        dict(zip(_CHUNK_COLS, row)) for row in cur.fetchall()
                    ]
            else:
                parents = []

            expanded: list[dict] = []
            if section_expansion and children:
                seen = {c["id"] for c in children}
                seen.update(p["id"] for p in parents)
                keys = {
                    (c["book_no"], c["reading_no"], c["module_no"])
                    for c in children
                    if c["module_no"] is not None
                }
                for book_no, reading_no, module_no in keys:
                    cur.execute(
                        f"""
                        SELECT {', '.join(_CHUNK_COLS)}
                        FROM frm_books
                        WHERE book_no = %s AND reading_no = %s AND module_no = %s
                        ORDER BY chunk_index
                        """,
                        (book_no, reading_no, module_no),
                    )
                    for row in cur.fetchall():
                        row_dict = dict(zip(_CHUNK_COLS, row))
                        if row_dict["id"] not in seen:
                            seen.add(row_dict["id"])
                            expanded.append(row_dict)
        return children + parents + expanded
    finally:
        conn.close()


def _filter_clause(filters: dict[str, Any]) -> tuple[str, list]:
    clauses: list[str] = []
    params: list[Any] = []
    for col in ("book_no", "year", "reading_no"):
        if filters.get(col) is not None:
            clauses.append(f"{col} = %s")
            params.append(filters[col])
    for col in ("module_no", "lo_code", "section"):
        if filters.get(col) is not None:
            clauses.append(f"{col} = %s")
            params.append(filters[col])
    if clauses:
        return " AND " + " AND ".join(clauses), params
    return "", params


def delete_all() -> int:
    conn = db.connect()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM frm_books")
            deleted = cur.rowcount
        conn.commit()
        return deleted
    finally:
        conn.close()