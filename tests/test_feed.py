import asyncio
import time

import pytest

from hypertrader.data.feeds import start_realtime_feed, QUEUE


@pytest.mark.asyncio
async def test_binance_feed():
    asyncio.create_task(start_realtime_feed(["BTC-USDT"]))
    t0 = time.time()
    tick = await asyncio.wait_for(QUEUE.get(), timeout=10)
    assert "price" in tick and time.time() - t0 < 10
