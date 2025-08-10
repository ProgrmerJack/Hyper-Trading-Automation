import asyncio
import pytest
import websockets

from hypertrader.feeds.private_ws import PrivateWebSocketFeed
from hypertrader.data.oms_store import OMSStore


class DummyWS:
    async def close(self):
        pass


class FakeResp:
    def __init__(self, data=None):
        self._data = data or {}

    def json(self):
        return self._data

    def raise_for_status(self):
        return None


@pytest.mark.asyncio
async def test_listen_key_keepalive(monkeypatch, tmp_path):
    class DummyClient:
        def __init__(self):
            self.calls = {"put": 0}

        async def post(self, url, headers=None):
            return FakeResp({"listenKey": "abc"})

        async def put(self, url, params=None, headers=None):
            self.calls["put"] += 1
            return FakeResp()

        async def delete(self, url, params=None, headers=None):
            return FakeResp()

        async def aclose(self):
            pass

    async def fake_connect(url):
        return DummyWS()

    store = OMSStore(tmp_path / "db.sqlite")
    feed = PrivateWebSocketFeed(
        "binance", store, "k", "s", listen_key_refresh=0.1
    )
    feed._http = DummyClient()
    monkeypatch.setattr(websockets, "connect", lambda url: fake_connect(url))

    await feed._connect()
    await asyncio.sleep(0.25)
    assert feed._http.calls["put"] >= 1
    await feed.close()
    await store.close()
