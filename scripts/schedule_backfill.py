"""Run question metadata backfill every five minutes with provider rotation."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.logging import setup_logging  # noqa: E402

setup_logging()

from app import db  # noqa: E402
from app.ingestion.question_ingestion import backfill_question_metadata  # noqa: E402
from app.llm import ingest_provider_specs  # noqa: E402


INTERVAL_SECONDS = 5 * 60
BATCH_SIZE = 3


def main() -> int:
    print("Starting question metadata scheduler: 3 questions every 5 minutes")
    provider_index = 0
    try:
        while True:
            try:
                if not db.is_ready():
                    print(f"Database not ready: {db.readiness_hint()}")
                else:
                    result = backfill_question_metadata(
                        limit=BATCH_SIZE,
                        provider_specs=ingest_provider_specs(),
                        provider_index=provider_index,
                    )
                    provider_index = result["next_provider_index"]
                    attempts = ", ".join(
                        f"qid={attempt['qid']} {attempt['provider']} {attempt['status']}"
                        for attempt in result["attempts"]
                    ) or "none"
                    print(
                        f"BACKFILL batch processed={result['processed']} "
                        f"updated={result['updated']} failed={result['failed']} "
                        f"attempts=[{attempts}] next_provider_index={provider_index}",
                        flush=True,
                    )
            except Exception as exc:
                print(f"Backfill cycle failed: {exc}")
            time.sleep(INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("Question metadata scheduler stopped")
        return 0


if __name__ == "__main__":
    sys.exit(main())
