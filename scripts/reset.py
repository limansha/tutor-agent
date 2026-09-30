"""Drop the frm_books table and re-apply sql/schema.sql (start fresh).

Run via: make reset
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.logging import setup_logging  # noqa: E402
setup_logging()

from app import db  # noqa: E402


def main() -> int:
    if not db.config.DATABASE_URL:
        print("ERROR: DATABASE_URL not set. Copy .env.example -> .env and fill it in.")
        return 1
    conn = db.connect()
    try:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS frm_books CASCADE")
        conn.commit()
    finally:
        conn.close()

    db.init_schema()
    print("Schema re-applied. Next: make ingest")
    return 0


if __name__ == "__main__":
    sys.exit(main())