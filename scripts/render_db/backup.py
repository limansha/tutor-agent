"""Create a local backup of a Render PostgreSQL database.

Usage:
    python scripts/render_db/backup.py 'postgresql://...'
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DUMP_DIR = REPO_ROOT / "render_db_dumps"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database_url", help="external Render PostgreSQL connection URL")
    args = parser.parse_args()
    database_url = args.database_url.strip()
    if shutil.which("pg_dump") is None:
        print("ERROR: pg_dump is not installed or not on PATH.")
        return 1

    DUMP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    dump_path = DUMP_DIR / f"tutor-agent-{timestamp}.dump"
    try:
        subprocess.run(
            [
                "pg_dump",
                "--dbname",
                database_url,
                "--format=custom",
                "--file",
                str(dump_path),
                "--no-owner",
                "--no-acl",
                "--verbose",
            ],
            check=True,
        )
        subprocess.run(
            ["pg_restore", "--list", str(dump_path)],
            check=True,
            stdout=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        dump_path.unlink(missing_ok=True)
        print(f"ERROR: backup failed: {exc}")
        return 1

    print(f"Backup created: {dump_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
