from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Optional, List

import contextlib
import requests
import websockets

from ..data.oms_store import OMSStore
from ..execution.ccxt_executor import cancel_all
from ..utils.monitoring import (
    listenkey_refresh_counter,
    ws_ping_counter,
    ws_pong_counter,
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
    ts: int           # epoch ms


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
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._listen_key: Optional[str] = None
        self._keepalive_task: Optional[asyncio.Task] = None

    async def _binance_listen_key(self) -> str:
        if self._listen_key:
            return self._listen_key
        if self.market == "futures":
            url = "https://fapi.binance.com/fapi/v1/listenKey"
        else:
            url = "https://api.binance.com/api/v3/userDataStream"
        resp = await asyncio.to_thread(
            requests.post, url, headers={"X-MBX-APIKEY": self.api_key}
        )
        resp.raise_for_status()
        self._listen_key = resp.json()["listenKey"]
        return self._listen_key

    async def _binance_keepalive(self) -> None:
        while self._ws is not None and self._listen_key:
            await asyncio.sleep(self.listen_key_refresh)
            if self.market == "futures":
                url = "https://fapi.binance.com/fapi/v1/listenKey"
            else:
                url = "https://api.binance.com/api/v3/userDataStream"
            try:
                await asyncio.to_thread(
                    requests.put,
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
                    await asyncio.to_thread(
                        requests.delete,
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
        status = msg.get("X")
        order_id = str(msg.get("i"))
        client_id = msg.get("c")
        side = msg.get("S")
        symbol = msg.get("s")
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
                ts=int(msg.get("E", 0)),
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
                    ts=int(msg.get("T", 0)),
                )
            )
        return evs

    def _map_binance_futures(self, event: dict) -> List[OrderEvent]:
        if event.get("e") != "ORDER_TRADE_UPDATE":
            return []
        o = event.get("o", {})
        status = o.get("X")
        order_id = str(o.get("i"))
        client_id = o.get("c")
        side = o.get("S")
        symbol = o.get("s")
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
                ts=int(event.get("E", 0)),
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
                    ts=int(o.get("T", 0)),
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
            await self.store.update_order_status(ev.order_id, ev.status)
            if ev.qty or ev.fee:
                await self.store.record_fill(
                    ev.order_id, ev.qty, ev.price, ev.fee, ev.ts / 1000
                )
            if self.market == "futures" and ev.qty:
                sign = 1 if ev.side == "BUY" else -1
                await self.store.upsert_position(
                    ev.symbol, sign * ev.qty, ev.price, None, ev.ts / 1000
                )

    async def run(self) -> None:
        backoff = 1
        while True:
            if self._ws is None:
                try:
                    await self._connect()
                    backoff = 1
                except Exception:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 30)
                    continue
            try:
                ping = self._ws.ping()
                ws_ping_counter.inc()
                await asyncio.wait_for(ping, timeout=self.heartbeat)
                ws_pong_counter.inc()
                raw = await asyncio.wait_for(self._ws.recv(), timeout=self.heartbeat)
                msg = json.loads(raw)
                if self.exchange == "binance":
                    await self._handle_binance(msg)
            except Exception:
                try:
                    if self._ws is not None:
                        await self._ws.close()
                finally:
                    self._ws = None
                await cancel_all()
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)

    async def close(self) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None
        if self._keepalive_task is not None:
            self._keepalive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._keepalive_task
            self._keepalive_task = None
