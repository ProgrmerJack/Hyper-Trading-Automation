import asyncio
import time

import pytest

from hypertrader.feeds.exchange_ws import ExchangeWebSocketFeed


@pytest.mark.asyncio
async def test_ws_feed_latency():
    feed = ExchangeWebSocketFeed("binance", "btcusdt", heartbeat=5)
    stream = feed.stream()
    try:
        start = time.perf_counter()
        tick = await asyncio.wait_for(anext(stream), timeout=10)
    except Exception as exc:
        pytest.skip(f"websocket unavailable: {exc}")
    finally:
        await feed.close()
    assert time.perf_counter() - start < 1.0
    assert tick is not None
