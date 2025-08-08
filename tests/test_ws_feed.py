import types
import pytest

from hypertrader.feeds.ccxt_ws import CCXTWebSocketFeed


@pytest.mark.asyncio
async def test_reconnect_on_error(monkeypatch):
    class DummyError(Exception):
        pass

    class DummyClient:
        def __init__(self, responses):
            self.responses = list(responses)
        async def watch_ticker(self, symbol):
            res = self.responses.pop(0)
            if isinstance(res, Exception):
                raise res
            return res
        async def close(self):
            pass

    clients = [
        DummyClient([{"price": 1}, DummyError("boom")]),
        DummyClient([{"price": 2}]),
    ]

    def binance(config):
        return clients.pop(0)

    dummy_ccxt = types.SimpleNamespace(binance=binance, BaseError=DummyError)
    monkeypatch.setattr("hypertrader.feeds.ccxt_ws.ccxt", dummy_ccxt)

    feed = CCXTWebSocketFeed("binance", "BTC/USDT", heartbeat=1)
    stream = feed.stream()
    first = await anext(stream)
    assert first["price"] == 1
    second = await anext(stream)
    assert second["price"] == 2
    await feed.close()
