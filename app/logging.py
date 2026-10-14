"""Logging setup: all log levels to stdout, so hosting platforms capture them."""

from __future__ import annotations

import logging
import sys


def setup_logging() -> None:
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG,
        format="[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
