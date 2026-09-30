"""CLI wrapper for the ingestion pipeline (parse -> chunk -> embed -> store).

Run via: make ingest  (optionally: --books-dir, --force-reingest)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.logging import setup_logging  # noqa: E402
setup_logging()

from app import db  # noqa: E402
from app.config import config  # noqa: E402
from app.ingestion.books_pipeline import ingest_books  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest FRM PDFs into pgvector.")
    parser.add_argument("--books-dir", default=config.BOOKS_DIR)
    parser.add_argument("--force-reingest", action="store_true")
    args = parser.parse_args()

    if not db.is_ready():
        print(f"ERROR: database not ready. Run `make db-setup` first. ({db.readiness_hint()})")
        return 1

    start = time.monotonic()
    results = ingest_books(args.books_dir, force_reingest=args.force_reingest)
    elapsed = int((time.monotonic() - start) * 1000)

    for r in results:
        print(
            f"[{r.status:<7}] {r.filename}: pages={r.pages} "
            f"parents={r.parents} children={r.children} tokens={r.tokens}"
            + (f" ({r.detail})" if r.detail else "")
        )

    print(
        f"\nTotal: {len(results)} files, "
        f"{sum(r.parents for r in results)} parents, "
        f"{sum(r.children for r in results)} children, "
        f"{sum(r.tokens for r in results)} tokens in {elapsed} ms"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())