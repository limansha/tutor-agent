"""Retry missing LLM-generated question fields; suitable for cron."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.logging import setup_logging  # noqa: E402

setup_logging()

from app import db  # noqa: E402
from app.ingestion.question_ingestion import backfill_question_metadata  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Fill missing question answer metadata.")
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()
    if not db.is_ready():
        print(f"ERROR: database not ready. ({db.readiness_hint()})")
        return 1
    result = backfill_question_metadata(args.limit)
    print(
        f"Processed {result['processed']}; updated {result['updated']}; "
        f"failed {result['failed']}"
    )
    return 0 if result["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
