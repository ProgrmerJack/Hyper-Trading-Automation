"""Structured logging utilities."""

from __future__ import annotations

import json
import logging
from typing import Any


def get_logger(name: str = "hypertrader") -> logging.Logger:
    """Return a logger configured to emit JSON formatted messages."""

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def log_json(logger: logging.Logger, event: str, **kwargs: Any) -> None:
    """Emit a structured JSON log entry."""

    payload = {"event": event, **kwargs}
    logger.info(json.dumps(payload))


__all__ = ["get_logger", "log_json"]

