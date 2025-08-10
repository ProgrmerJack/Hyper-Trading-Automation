import asyncio
import time
import pytest

from hypertrader.data.oms_store import OMSStore


@pytest.mark.asyncio
async def test_oms_store_roundtrip(tmp_path) -> None:
    db = tmp_path / "orders.db"
    store = OMSStore(db)
    now = time.time()
    await store.record_order("1", "c1", "BTC/USDT", "buy", 1.0, 100.0, "open", now)
    rows = list(await store.fetch_open_orders())
    assert len(rows) == 1
    assert rows[0][0] == "1"
    assert rows[0][1] == "BTC/USDT"
    await store.update_order_status("1", "FILLED")
    assert list(await store.fetch_open_orders()) == []
    await store.record_fill("1", 0.5, 100.0, 0.01, now)
    await store.upsert_position("BTC/USDT", 0.5, 100.0, None, now)
    pos = list(await store.fetch_positions())
    assert pos[0][0] == "BTC/USDT"
    await store.close()
