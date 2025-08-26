"""Network helpers such as retry wrappers."""

from __future__ import annotations

import time
from typing import Any, Callable, TypeVar


T = TypeVar("T")


def fetch_with_retry(
    func: Callable[..., T],
    *args: Any,
    retries: int = 3,
    delay: float = 1.0,
    **kwargs: Any,
) -> T:
    """Execute ``func`` with retry logic."""

    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - defensive
            last_exc = exc
            if attempt == retries:
                raise
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc


__all__ = ["fetch_with_retry"]

