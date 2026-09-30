"""CLI wrapper for the question ingestion pipeline (extract → LLM → store).

Run via: make ingest-questions  (optionally: --books-dir, --force-reingest)
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
from app.ingestion.question_ingestion import ingest_questions, store_questions_from_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest FRM questions into frm_questions table.")
    parser.add_argument("--books-dir", default=config.QUESTIONS_DIR)
    parser.add_argument("--force-reingest", action="store_true")
    parser.add_argument("--phase", choices=["1", "store"], default="1",
                        help="Phase 1: text extraction + LLM + JSON dump. "
                             "phase store: load questions_*.json into DB.")
    args = parser.parse_args()

    if not db.is_ready():
        print(f"ERROR: database not ready. Run `make db-setup` first. ({db.readiness_hint()})")
        return 1

    start = time.monotonic()

    if args.phase == "1":
        results = ingest_questions(args.books_dir)
        elapsed = int((time.monotonic() - start) * 1000)
        print(
            f"[{results['status']:<7}] {results.get('total_files', 0)} files, "
            f"{results.get('total_questions', 0)} questions "
            f"in {elapsed} ms"
        )
        if results.get("detail"):
            print(f"  ({results['detail']})")
        if results.get("message"):
            print(f"\n{results['message']}")
    elif args.phase == "store":
        results = store_questions_from_json(force_reingest=args.force_reingest)
        elapsed = int((time.monotonic() - start) * 1000)
        print(
            f"[{results['status']:<7}] {results.get('total_files', 0)} files, "
            f"{results.get('total_questions', 0)} questions "
            f"in {elapsed} ms"
        )
        if results.get("detail"):
            print(f"  ({results['detail']})")
        if results.get("message"):
            print(f"\n{results['message']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
