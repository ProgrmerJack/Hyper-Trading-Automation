"""Structured logging utilities."""

from __future__ import annotations

import json
import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Any
from datetime import datetime

# Optional numeric libs for safe JSON encoding
try:  # pragma: no cover - availability depends on environment
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:  # pragma: no cover - availability depends on environment
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore


def get_logger(name: str = "hypertrader") -> logging.Logger:
    """Return a logger configured to emit JSON formatted messages.

    If the environment variable ``LOG_FILE`` is set, logs are also written to a
    rotating file handler with size and backup limits controlled by
    ``LOG_MAX_BYTES`` and ``LOG_BACKUP_COUNT``. The console handler always
    emits plain JSON lines suitable for log ingestion.
    """

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, level, logging.INFO))

    # Console JSON handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stream_handler)

    # Optional rotating file handler
    log_path = os.getenv("LOG_FILE")
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        max_bytes = int(os.getenv("LOG_MAX_BYTES", "10485760"))  # 10 MB
        backup_count = int(os.getenv("LOG_BACKUP_COUNT", "5"))
        file_handler = RotatingFileHandler(log_path, maxBytes=max_bytes, backupCount=backup_count)
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(file_handler)

    return logger


def _json_default(o: Any):
    """Best-effort JSON serializer for numpy/pandas/datetime and similar types."""
    try:
        # Numpy scalars
        if np is not None and isinstance(o, getattr(np, "generic", ())) :
            return o.item()
        # Numpy arrays
        if hasattr(o, "tolist"):
            return o.tolist()
        # Pandas Timestamp
        if pd is not None and isinstance(o, getattr(pd, "Timestamp", ())):
            return o.isoformat()
        # Datetime
        if isinstance(o, datetime):
            return o.isoformat()
        # Sets
        if isinstance(o, (set, frozenset)):
            return list(o)
    except Exception:
        pass
    # Fallback to string representation
    return str(o)


def log_json(logger: logging.Logger, event: str, **kwargs: Any) -> None:
    """Emit a structured JSON log entry."""

    payload = {"event": event, **kwargs}
    logger.info(json.dumps(payload, default=_json_default))


__all__ = ["get_logger", "log_json"]

