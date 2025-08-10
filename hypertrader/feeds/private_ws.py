from __future__ import annotations

import asyncio
import json
from typing import Optional

import websockets
import requests

from ..data.oms_store import OMSStore
from ..execution.ccxt_executor import cancel_all


class PrivateWebSocketFeed:
    """User-data stream for account events.

    Currently supports Binance.  Events are written into ``OMSStore`` in
    real time.  On disconnect the feed attempts to cancel all open orders to
    avoid running blind.
    """

    def __init__(self, exchange: str, store: OMSStore, api_key: str, api_secret: str, heartbeat: int = 30) -> None:
        self.exchange = exchange.lower()
        self.store = store
        self.api_key = api_key
        self.api_secret = api_secret
        self.heartbeat = heartbeat
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._listen_key: Optional[str] = None

    async def _binance_listen_key(self) -> str:
        if self._listen_key:
            return self._listen_key
        url = "https://api.binance.com/api/v3/userDataStream"
        resp = await asyncio.to_thread(requests.post, url, headers={"X-MBX-APIKEY": self.api_key})
        resp.raise_for_status()
        self._listen_key = resp.json()["listenKey"]
        return self._listen_key

    async def _connect(self) -> None:
        if self.exchange == "binance":
            key = await self._binance_listen_key()
            url = f"wss://stream.binance.com:9443/ws/{key}"
            self._ws = await websockets.connect(url)
        else:
            raise ValueError("unsupported exchange for private stream")

    async def _handle_binance(self, msg: dict) -> None:
        if msg.get("e") != "executionReport":
            return
        order_id = msg.get("c") or str(msg.get("i"))
        status = msg.get("X")
        if order_id and status:
            await self.store.update_order_status(order_id, status)
        if msg.get("x") == "TRADE":
            qty = float(msg.get("l", 0))
            price = float(msg.get("L", 0))
            fee = float(msg.get("n", 0))
            ts = msg.get("T", 0) / 1000
            await self.store.record_fill(order_id, qty, price, fee, ts)

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
