"""Database access layer (Postgres + pgvector).

The operations here are the DB "placeholders" for the plan: they are fully
written out, but nothing connects or mutates Postgres until YOU run the make
targets (`make db-setup`, `make ingest`, `make run`). Every function that
touches the DB guards on `is_ready()`; endpoints surface a 503 with a hint
when the schema has not been initialized.
"""

from __future__ import annotations

from app.config import config


class DatabaseNotReady(RuntimeError):
    """Raised when DATABASE_URL is unset or the schema has not been created."""


def get_dsn() -> str:
    if not config.DATABASE_URL:
        raise DatabaseNotReady(
            "DATABASE_URL is not set. Copy .env.example to .env and fill it in, "
            "then run `make db-setup`."
        )
    return config.DATABASE_URL


def connect():
    """Open a psycopg connection. Caller is responsible for closing it."""
    import psycopg

    return psycopg.connect(get_dsn())


def is_ready() -> bool:
    """True when the pgvector schema exists and is reachable."""
    try:
        conn = connect()
    except Exception:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT to_regclass('public.frm_books')
                """
            )
            if cur.fetchone()[0] is None:
                return False
            cur.execute(
                """
                SELECT to_regclass('public.frm_questions')
                """
            )
            return cur.fetchone()[0] is not None
    except Exception:
        return False
    finally:
        conn.close()


def init_schema() -> None:
    """Apply sql/schema.sql (pgvector extension, frm_books table, HNSW index)."""
    if not config.SCHEMA_SQL.exists():
        raise FileNotFoundError(f"schema file missing: {config.SCHEMA_SQL}")
    conn = connect()
    try:
        ddl = config.SCHEMA_SQL.read_text()
        with conn.cursor() as cur:
            cur.execute(ddl)
        conn.commit()
    finally:
        conn.close()


def readiness_hint() -> str:
    return (
        "Postgres/pgvector is not ready. Run: make env && make db-setup && make ingest"
    )