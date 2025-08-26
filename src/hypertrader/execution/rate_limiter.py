"""Simple asynchronous token bucket rate limiter."""
from __future__ import annotations

import asyncio
import logging
import time

from ..utils.monitoring import rate_limit_throttle_counter


class TokenBucket:
    """Token bucket limiting the rate of operations.

    Parameters
    ----------
    rate:
        Number of tokens added per second.
    capacity:
        Maximum number of tokens in the bucket (burst size).
    """

    def __init__(self, rate: float, capacity: int) -> None:
        self.rate = float(rate)
        self.capacity = int(capacity)
        self.tokens = float(capacity)
        self.updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: float = 1.0) -> None:
        """Consume ``tokens`` waiting if necessary."""
        while True:
            async with self._lock:
                now = time.monotonic()
                elapsed = now - self.updated
                self.updated = now
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return
                wait = (tokens - self.tokens) / self.rate
                rate_limit_throttle_counter.inc()
                logging.info("rate limiter throttling for %.2fs", wait)
            await asyncio.sleep(wait)


__all__ = ["TokenBucket"]

