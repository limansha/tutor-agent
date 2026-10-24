"""Load successful question JSON files into the database."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.logging import setup_logging  # noqa: E402

setup_logging()
sys.stdout.reconfigure(line_buffering=True)

from app import db  # noqa: E402
from app.ingestion.question_ingestion import store_questions_from_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Store question JSON files in frm_questions.")
    parser.add_argument("--json-dir", default=None)
    args = parser.parse_args()

    if not db.is_ready():
        print(f"ERROR: database not ready. Run `make db-setup` first. ({db.readiness_hint()})")
        return 1

    start = time.monotonic()
    results = store_questions_from_json(json_dir=args.json_dir)
    elapsed = int((time.monotonic() - start) * 1000)
    print(
        f"[{results['status']:<7}] {results.get('total_files', 0)} files, "
        f"{results.get('total_questions', 0)} questions in {elapsed} ms"
    )
    if results.get("message"):
        print(f"\n{results['message']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
