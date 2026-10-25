"""Run Gemini question metadata backfill every three minutes."""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.logging import setup_logging  # noqa: E402

setup_logging()

from app import db  # noqa: E402
from app.config import config  # noqa: E402
from app.ingestion.question_ingestion import backfill_question_metadata  # noqa: E402


INTERVAL_SECONDS = 3 * 60
BATCH_SIZE = 3


def main() -> int:
    print(
        "Starting Gemini question metadata scheduler: "
        f"3 questions every 3 minutes; key_configured={bool(config.GEMINI_API_KEY)}; "
        f"model={config.GEMINI_MODEL}",
        flush=True,
    )
    try:
        while True:
            try:
                if not db.is_ready():
                    print(f"Database not ready: {db.readiness_hint()}")
                else:
                    result = backfill_question_metadata(
                        limit=BATCH_SIZE,
                    )
                    attempts = ", ".join(
                        f"qid={attempt['qid']} {attempt['provider']} {attempt['status']}"
                        for attempt in result["attempts"]
                    ) or "none"
                    print(
                        f"BACKFILL batch processed={result['processed']} "
                        f"updated={result['updated']} failed={result['failed']} "
                        f"attempts=[{attempts}]",
                        flush=True,
                    )
            except Exception as exc:
                print(f"Backfill cycle failed: {exc}")
                traceback.print_exc()
            time.sleep(INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("Question metadata scheduler stopped")
        return 0


if __name__ == "__main__":
    sys.exit(main())
