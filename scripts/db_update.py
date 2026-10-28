"""Apply incremental database updates.

Run via: make db-update
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import db  # noqa: E402


MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "sql" / "migrations"


def main() -> int:
    if not db.config.DATABASE_URL:
        print("ERROR: DATABASE_URL not set. Copy .env.example -> .env and fill it in.")
        return 1
    migrations = sorted(MIGRATIONS_DIR.glob("*.sql"))
    if not migrations:
        print(f"ERROR: no migration files found in {MIGRATIONS_DIR}")
        return 1

    with db.connect() as conn:
        with conn.cursor() as cur:
            for migration in migrations:
                cur.execute(migration.read_text())
        conn.commit()
    print(f"Applied {len(migrations)} database update(s) from {MIGRATIONS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
