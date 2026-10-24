"""CLI wrapper for the question JSON ingestion pipeline.

Run via: make ingest-questions-json.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.logging import setup_logging  # noqa: E402
setup_logging()
sys.stdout.reconfigure(line_buffering=True)  # logs appear promptly when piped

from app.config import config  # noqa: E402
from app.ingestion.question_ingestion import ingest_questions, process_parsed_text_to_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract FRM questions into JSON files.")
    parser.add_argument("--books-dir", default=config.QUESTIONS_DIR)
    parser.add_argument("--phase", choices=["1", "parse"], default="1",
                        help="Phase 1: text extraction + LLM + JSON dump. "
                             "parse: process parsed_text_files/*.txt in batches via LLM -> json_files.")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    start = time.monotonic()

    if args.phase == "1":
        results = ingest_questions(args.books_dir, batch_size=args.batch_size)
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
    else:
        parsed_dir = Path(config.QUESTIONS_PARSED_DIR)
        for p in sorted(parsed_dir.glob("*.txt")):
            print(f"Parsing {p.name}")
            res = process_parsed_text_to_json(p, batch_size=args.batch_size)
            print(res)
        elapsed = int((time.monotonic() - start) * 1000)
        print(f"Done in {elapsed} ms")

    return 0


if __name__ == "__main__":
    sys.exit(main())
