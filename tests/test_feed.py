import asyncio

import pytest

from hypertrader.data.feeds import start_realtime_feed, QUEUE


@pytest.mark.asyncio
async def test_binance_feed(monkeypatch):
    while not QUEUE.empty():
        QUEUE.get_nowait()

    class DummyExchange:
        def __init__(self, *args, **kwargs):
            pass

    class DummyFeedHandler:
        def __init__(self, *args, **kwargs):
            pass

        def add_feed(self, feed):
            pass

        def run(self):
            asyncio.run(QUEUE.put({"price": 1}))

        def stop(self):
            pass

    monkeypatch.setattr("hypertrader.data.feeds.FeedHandler", DummyFeedHandler)
    monkeypatch.setattr("hypertrader.data.feeds.Binance", DummyExchange)
    monkeypatch.setattr("hypertrader.data.feeds.Coinbase", DummyExchange)
    monkeypatch.setattr("hypertrader.data.feeds.Kraken", DummyExchange)

    task = asyncio.create_task(start_realtime_feed(["BTC-USDT"]))
    tick = await asyncio.wait_for(QUEUE.get(), timeout=1)
    assert tick["price"] == 1
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
