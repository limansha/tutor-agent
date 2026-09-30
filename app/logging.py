"""Logging setup: writes all log levels to timestamped files in logs/."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"


def setup_logging() -> None:
    if any(isinstance(h, logging.FileHandler) for h in logging.getLogger().handlers):
        return
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        filename=str(LOGS_DIR / f"app_{ts}.log"),
        level=logging.DEBUG,
        format="[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
