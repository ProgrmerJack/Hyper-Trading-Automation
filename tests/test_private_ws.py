import asyncio
import time

import pytest
import requests
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
    calls = {"put": 0}

    def fake_post(url, headers):
        return FakeResp({"listenKey": "abc"})

    def fake_put(url, params, headers):
        calls["put"] += 1
        return FakeResp()

    async def fake_connect(url):
        return DummyWS()

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(requests, "put", fake_put)
    monkeypatch.setattr(websockets, "connect", lambda url: fake_connect(url))

    store = OMSStore(tmp_path / "db.sqlite")
    feed = PrivateWebSocketFeed(
        "binance", store, "k", "s", listen_key_refresh=0.1
    )
    await feed._connect()
    await asyncio.sleep(0.25)
    assert calls["put"] >= 1
    await feed.close()
    await store.close()
