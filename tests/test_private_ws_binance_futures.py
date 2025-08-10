import json
import asyncio

import pytest

from hypertrader.feeds.private_ws import PrivateWebSocketFeed
from hypertrader.data.oms_store import OMSStore


def load_event():
    with open("tests/fixtures/binance_futures_events.json") as f:
        return json.load(f)


@pytest.mark.asyncio
async def test_map_binance_futures():
    store = OMSStore(":memory:")
    feed = PrivateWebSocketFeed("binance", store, "k", "s", market="futures")
    evs = feed._map_binance_futures(load_event())
    assert len(evs) == 2
    status_ev, fill_ev = evs
    assert status_ev.status == "FILLED"
    assert fill_ev.qty == pytest.approx(0.001)
    assert fill_ev.fee == pytest.approx(0.00001)
    await store.close()


@pytest.mark.asyncio
async def test_store_updates_from_futures_event(tmp_path):
    store = OMSStore(tmp_path / "db.sqlite")
    feed = PrivateWebSocketFeed("binance", store, "k", "s", market="futures")
    await store.record_order("123456789", "myClientId", "BTCUSDT", "BUY", 0.001, 10000, "NEW", 0)
    await feed._handle_binance(load_event())
    cur = store.conn.execute("SELECT status FROM orders WHERE id=?", ("123456789",))
    assert cur.fetchone()[0] == "FILLED"
    cur = store.conn.execute("SELECT qty, price, fee FROM fills WHERE order_id=?", ("123456789",))
    qty, price, fee = cur.fetchone()
    assert qty == pytest.approx(0.001)
    assert price == pytest.approx(10000)
    assert fee == pytest.approx(0.00001)
    await store.close()


@pytest.mark.asyncio
async def test_partial_fill_sequence(tmp_path):
    store = OMSStore(tmp_path / "db.sqlite")
    feed = PrivateWebSocketFeed("binance", store, "k", "s", market="futures")
    await store.record_order("1", "cid", "BTCUSDT", "BUY", 0.002, 10000, "NEW", 0)
    partial = {
        "e": "ORDER_TRADE_UPDATE",
        "E": 1000,
        "o": {
            "s": "BTCUSDT",
            "c": "cid",
            "i": 1,
            "X": "PARTIALLY_FILLED",
            "S": "BUY",
            "q": "0.002",
            "p": "10000",
            "l": "0.001",
            "L": "10000",
            "n": "0.00001",
            "T": 1100,
            "x": "TRADE",
        },
    }
    final = {
        "e": "ORDER_TRADE_UPDATE",
        "E": 2000,
        "o": {
            "s": "BTCUSDT",
            "c": "cid",
            "i": 1,
            "X": "FILLED",
            "S": "BUY",
            "q": "0.002",
            "p": "10000",
            "l": "0.001",
            "L": "10000",
            "n": "0.00001",
            "T": 2100,
            "x": "TRADE",
        },
    }
    await feed._handle_binance(partial)
    await feed._handle_binance(final)
    cur = store.conn.execute("SELECT status FROM orders WHERE id=?", ("1",))
    assert cur.fetchone()[0] == "FILLED"
    cur = store.conn.execute("SELECT SUM(qty) FROM fills WHERE order_id=?", ("1",))
    assert cur.fetchone()[0] == pytest.approx(0.002)
    await store.close()


@pytest.mark.asyncio
async def test_duplicate_event_idempotent(tmp_path):
    store = OMSStore(tmp_path / "db.sqlite")
    feed = PrivateWebSocketFeed("binance", store, "k", "s", market="futures")
    await store.record_order("123456789", "cid", "BTCUSDT", "BUY", 0.001, 10000, "NEW", 0)
    ev = load_event()
    await feed._handle_binance(ev)
    await feed._handle_binance(ev)
    cur = store.conn.execute("SELECT COUNT(*) FROM fills WHERE order_id=?", ("123456789",))
    assert cur.fetchone()[0] == 1
    await store.close()
