import asyncio
import time

import pytest

from hypertrader.execution.rate_limiter import TokenBucket


@pytest.mark.asyncio
async def test_token_bucket_waits():
    bucket = TokenBucket(rate=1, capacity=1)
    await bucket.acquire()
    start = time.perf_counter()
    await bucket.acquire()
    elapsed = time.perf_counter() - start
    assert elapsed >= 0.9


@pytest.mark.asyncio
async def test_token_bucket_throttles_burst():
    bucket = TokenBucket(rate=5, capacity=5)
    start = time.perf_counter()
    for _ in range(10):
        await bucket.acquire()
    elapsed = time.perf_counter() - start
    assert elapsed >= 1.0
