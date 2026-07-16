"""Project-wide logger factory.

Usage::

    from m5forecast.utils.logging import get_logger
    log = get_logger(__name__)
    log.info("panel built: %s rows", len(panel))
"""

from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_configured = False


def get_logger(name: str) -> logging.Logger:
    """Return a logger writing to stdout with the project format."""
    global _configured
    if not _configured:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%H:%M:%S"))
        root = logging.getLogger("m5forecast")
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        root.propagate = False
        _configured = True
    return logging.getLogger(name if name.startswith("m5forecast") else f"m5forecast.{name}")
