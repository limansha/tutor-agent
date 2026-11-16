"""Restore a local Render PostgreSQL dump into a new Render database.

Usage:
    TARGET_DATABASE_URL='postgresql://...' \
        python scripts/render_db/restore.py tutor-agent-20261024-120000.dump --confirm
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DUMP_DIR = REPO_ROOT / "render_db_dumps"


def _dump_path(value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = DUMP_DIR / candidate
    candidate = candidate.resolve()
    if DUMP_DIR.resolve() not in candidate.parents:
        raise argparse.ArgumentTypeError(
            f"dump must be inside {DUMP_DIR}"
        )
    if not candidate.is_file():
        raise argparse.ArgumentTypeError(f"dump does not exist: {candidate}")
    return candidate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump", type=_dump_path, help="dump filename in render_db_dumps/")
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="confirm that TARGET_DATABASE_URL points to the new empty database",
    )
    args = parser.parse_args()

    database_url = os.environ.get("TARGET_DATABASE_URL", "").strip()
    if not database_url:
        print("ERROR: set TARGET_DATABASE_URL to the new Render external database URL.")
        return 1
    if not args.confirm:
        print("ERROR: pass --confirm after checking the target is the new empty database.")
        return 1
    if shutil.which("pg_restore") is None:
        print("ERROR: pg_restore is not installed or not on PATH.")
        return 1

    try:
        subprocess.run(
            [
                "pg_restore",
                "--dbname",
                database_url,
                "--no-owner",
                "--no-acl",
                "--exit-on-error",
                "--single-transaction",
                str(args.dump),
            ],
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: restore failed: {exc}")
        return 1

    print(f"Restore completed from: {args.dump}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
