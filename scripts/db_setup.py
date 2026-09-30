"""Apply sql/schema.sql to the database from DATABASE_URL (.env).

Run via: make db-setup
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
    print(f"Applying schema: {db.config.SCHEMA_SQL}")
    db.init_schema()
    print("Schema applied. Next: make ingest")
    return 0


if __name__ == "__main__":
    sys.exit(main())