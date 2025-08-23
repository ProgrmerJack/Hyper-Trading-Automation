from __future__ import annotations

import asyncio
import json
import time
import random
from dataclasses import dataclass
from typing import Optional, List

import contextlib
import httpx
import websockets
from websockets.asyncio.client import ClientConnection

from ..data.oms_store import OMSStore
from ..execution.ccxt_executor import cancel_all
from ..utils.monitoring import (
    ack_fill_histogram,
    decision_ack_histogram,
    listenkey_refresh_counter,
    ws_ping_counter,
    ws_pong_counter,
    ws_ping_rtt_histogram,
    ws_reconnect_counter,
)


@dataclass
class OrderEvent:
    venue: str
    market_type: str  # 'spot' | 'futures'
    symbol: str
    order_id: str
    client_id: str | None
    status: str       # 'NEW'|'PARTIALLY_FILLED'|'FILLED'|'CANCELED'|'REJECTED'
    side: str         # 'BUY'|'SELL'
    qty: float
    price: float
    fee: float
    ts: float         # epoch seconds


class PrivateWebSocketFeed:
    """User-data stream for account events.

    Currently supports Binance.  Events are written into ``OMSStore`` in
    real time.  On disconnect the feed attempts to cancel all open orders to
    avoid running blind.
    """

    def __init__(
        self,
        exchange: str,
        store: OMSStore,
        api_key: str,
        api_secret: str,
        heartbeat: int = 30,
        market: str = "spot",
        listen_key_refresh: int = 1500,
    ) -> None:
        self.exchange = exchange.lower()
        self.store = store
        self.api_key = api_key
        self.api_secret = api_secret
        self.heartbeat = heartbeat
        self.market = market.lower()
        self.listen_key_refresh = listen_key_refresh
        self._ws: Optional[ClientConnection] = None
        self._listen_key: Optional[str] = None
        self._keepalive_task: Optional[asyncio.Task] = None
        transport = httpx.AsyncHTTPTransport(retries=3)
        self._http = httpx.AsyncClient(timeout=5.0, transport=transport)
        self._ack_times: dict[str, float] = {}

    async def _binance_listen_key(self) -> str:
        if self._listen_key:
            return self._listen_key
        if self.market == "futures":
            url = "https://fapi.binance.com/fapi/v1/listenKey"
        else:
            url = "https://api.binance.com/api/v3/userDataStream"
        resp = await self._http.post(url, headers={"X-MBX-APIKEY": self.api_key})
        resp.raise_for_status()
        listen_key = resp.json()["listenKey"]
        self._listen_key = listen_key
        return listen_key

    async def _binance_keepalive(self) -> None:
        while self._ws is not None and self._listen_key:
            await asyncio.sleep(self.listen_key_refresh)
            if self.market == "futures":
                url = "https://fapi.binance.com/fapi/v1/listenKey"
            else:
                url = "https://api.binance.com/api/v3/userDataStream"
            try:
                await self._http.put(
                    url,
                    params={"listenKey": self._listen_key},
                    headers={"X-MBX-APIKEY": self.api_key},
                )
                listenkey_refresh_counter.inc()
            except Exception:
                break

    async def _connect(self) -> None:
        if self.exchange == "binance":
            if self._listen_key:
                # close old listen key
                if self.market == "futures":
                    url = "https://fapi.binance.com/fapi/v1/listenKey"
                else:
                    url = "https://api.binance.com/api/v3/userDataStream"
                try:
                    await self._http.delete(
                        url,
                        params={"listenKey": self._listen_key},
                        headers={"X-MBX-APIKEY": self.api_key},
                    )
                except Exception:
                    pass
                self._listen_key = None
            key = await self._binance_listen_key()
            if self.market == "futures":
                url = f"wss://fstream.binance.com/ws/{key}"
            else:
                url = f"wss://stream.binance.com:9443/ws/{key}"
            self._ws = await websockets.connect(url)
            ws_reconnect_counter.inc()
            if self._keepalive_task is None or self._keepalive_task.done():
                self._keepalive_task = asyncio.create_task(self._binance_keepalive())
        else:
            raise ValueError("unsupported exchange for private stream")

    def _map_binance_spot(self, msg: dict) -> List[OrderEvent]:
        if msg.get("e") != "executionReport":
            return []
        status = str(msg.get("X"))
        order_id = str(msg.get("i"))
        client_id = msg.get("c")
        side = str(msg.get("S"))
        symbol = str(msg.get("s"))
        evs: List[OrderEvent] = [
            OrderEvent(
                venue="binance",
                market_type="spot",
                symbol=symbol,
                order_id=order_id,
                client_id=client_id,
                status=status,
                side=side,
                qty=0.0,
                price=0.0,
                fee=0.0,
                ts=float(msg.get("E", 0)) / 1000,
            )
        ]
        if msg.get("x") == "TRADE" and float(msg.get("l", 0)) > 0:
            evs.append(
                OrderEvent(
                    venue="binance",
                    market_type="spot",
                    symbol=symbol,
                    order_id=order_id,
                    client_id=client_id,
                    status="FILLED" if status == "FILLED" else "PARTIALLY_FILLED",
                    side=side,
                    qty=float(msg.get("l", 0.0)),
                    price=float(msg.get("L", 0.0)),
                    fee=float(msg.get("n", 0.0)),
                    ts=float(msg.get("T", 0)) / 1000,
                )
            )
        return evs

    def _map_binance_futures(self, event: dict) -> List[OrderEvent]:
        if event.get("e") != "ORDER_TRADE_UPDATE":
            return []
        o = event.get("o", {})
        status = str(o.get("X"))
        order_id = str(o.get("i") or event.get("i"))
        client_id = o.get("c")
        side = str(o.get("S"))
        symbol = str(o.get("s"))
        evs: List[OrderEvent] = [
            OrderEvent(
                venue="binance",
                market_type="futures",
                symbol=symbol,
                order_id=order_id,
                client_id=client_id,
                status=status,
                side=side,
                qty=0.0,
                price=0.0,
                fee=0.0,
                ts=float(event.get("E", 0)) / 1000,
            )
        ]
        if float(o.get("l", 0)) > 0:
            evs.append(
                OrderEvent(
                    venue="binance",
                    market_type="futures",
                    symbol=symbol,
                    order_id=order_id,
                    client_id=client_id,
                    status="FILLED" if status == "FILLED" else "PARTIALLY_FILLED",
                    side=side,
                    qty=float(o.get("l", 0.0)),
                    price=float(o.get("L", 0.0)),
                    fee=float(o.get("n", 0.0)),
                    ts=float(o.get("T", 0)) / 1000,
                )
            )
        return evs

    async def _handle_binance(self, msg: dict) -> None:
        evs: List[OrderEvent] = []
        if self.market == "spot":
            evs = self._map_binance_spot(msg)
        elif self.market == "futures":
            evs = self._map_binance_futures(msg)
        for ev in evs:
            order_ts = await self.store.fetch_order_ts(ev.order_id)
            if order_ts is not None:
                if ev.qty == 0:
                    decision_ack_histogram.observe(ev.ts - order_ts)
                    self._ack_times[ev.order_id] = ev.ts
                else:
                    ack_ts = self._ack_times.get(ev.order_id, order_ts)
                    ack_fill_histogram.observe(ev.ts - ack_ts)
            await self.store.update_order_status(ev.order_id, ev.status)
            if ev.qty or ev.fee:
                await self.store.record_fill(
                    ev.order_id, ev.qty, ev.price, ev.fee, ev.ts
                )
            if self.market == "futures" and ev.qty:
                sign = 1 if ev.side == "BUY" else -1
                await self.store.upsert_position(
                    ev.symbol, sign * ev.qty, ev.price, None, ev.ts
                )
            if ev.status in ("FILLED", "CANCELED"):
                self._ack_times.pop(ev.order_id, None)

    async def run(self) -> None:
        backoff = 1
        missed = 0
        while True:
            if self._ws is None:
                try:
                    await self._connect()
                    backoff = 1
                except Exception:
                    await asyncio.sleep(backoff + random.uniform(0, backoff))
                    backoff = min(backoff * 2, 30)
                    continue
            try:
                ws = self._ws
                if ws is None:
                    continue
                start = time.perf_counter()
                ping = ws.ping()
                ws_ping_counter.inc()
                await asyncio.wait_for(ping, timeout=self.heartbeat)
                ws_pong_counter.inc()
                ws_ping_rtt_histogram.observe(time.perf_counter() - start)
                raw = await asyncio.wait_for(ws.recv(), timeout=self.heartbeat)
                missed = 0
                msg = json.loads(raw)
                if self.exchange == "binance":
                    await self._handle_binance(msg)
            except asyncio.TimeoutError:
                missed += 1
                if missed < 3:
                    continue
                try:
                    if self._ws is not None:
                        await self._ws.close()
                finally:
                    self._ws = None
                await cancel_all()
                await asyncio.sleep(backoff + random.uniform(0, backoff))
                backoff = min(backoff * 2, 30)
            except Exception:
                try:
                    if self._ws is not None:
                        await self._ws.close()
                finally:
                    self._ws = None
                await cancel_all()
                await asyncio.sleep(backoff + random.uniform(0, backoff))
                backoff = min(backoff * 2, 30)

    async def close(self) -> None:
        ws = self._ws
        if ws is not None:
            await ws.close()
            self._ws = None
        if self._keepalive_task is not None:
            self._keepalive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._keepalive_task
            self._keepalive_task = None
        await self._http.aclose()
