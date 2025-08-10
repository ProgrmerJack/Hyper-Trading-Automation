from __future__ import annotations

import asyncio
import json
import time
import random
from typing import AsyncIterator, Dict, Optional

import websockets

from ..utils.monitoring import (
    ws_ping_counter,
    ws_pong_counter,
    ws_ping_rtt_histogram,
    ws_reconnect_counter,
)


class ExchangeWebSocketFeed:
    """Minimal direct exchange WebSocket feed without paid dependencies.

    Parameters
    ----------
    exchange : str
        Exchange identifier (``"binance"`` or ``"bybit"``).
    symbol : str
        Trading pair symbol.  For Binance use ``"btcusdt"`` format, for
        Bybit use ``"BTCUSDT"``.
    heartbeat : int, optional
        Seconds to wait for a message before reconnecting.  Defaults to 30.
    """

    def __init__(self, exchange: str, symbol: str, heartbeat: int = 30) -> None:
        self.exchange = exchange.lower()
        self.symbol = symbol
        self.heartbeat = heartbeat
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._last_msg = time.time()

    async def _connect(self) -> None:
        if self.exchange == "binance":
            url = f"wss://stream.binance.com:9443/ws/{self.symbol.lower()}@ticker"
            self._ws = await websockets.connect(url)
        elif self.exchange == "bybit":
            url = "wss://stream.bybit.com/v5/public/spot"
            self._ws = await websockets.connect(url)
            sub = {"op": "subscribe", "args": [f"tickers.{self.symbol.upper()}"]}
            await self._ws.send(json.dumps(sub))
        else:
            raise ValueError("unsupported exchange")

    async def stream(self) -> AsyncIterator[Dict]:
        """Yield ticker messages indefinitely with automatic reconnection.

        When the heartbeat is missed the feed reconnects and yields ``None`` to
        signal a disconnect event to the caller.
        """

        backoff = 1
        missed = 0
        while True:
            if self._ws is None:
                try:
                    await self._connect()
                    backoff = 1
                    ws_reconnect_counter.inc()
                except Exception:
                    await asyncio.sleep(backoff + random.uniform(0, backoff))
                    backoff = min(backoff * 2, 30)
                    continue
            try:
                start = time.perf_counter()
                ping = self._ws.ping()
                ws_ping_counter.inc()
                await asyncio.wait_for(ping, timeout=self.heartbeat)
                ws_pong_counter.inc()
                ws_ping_rtt_histogram.observe(time.perf_counter() - start)
                msg = await asyncio.wait_for(self._ws.recv(), timeout=self.heartbeat)
                self._last_msg = time.time()
                missed = 0
                yield json.loads(msg)
            except asyncio.TimeoutError:
                missed += 1
                if missed < 3:
                    continue
                try:
                    if self._ws is not None:
                        await self._ws.close()
                finally:
                    self._ws = None
                yield None
                await asyncio.sleep(backoff + random.uniform(0, backoff))
                backoff = min(backoff * 2, 30)
            except Exception:
                try:
                    if self._ws is not None:
                        await self._ws.close()
                finally:
                    self._ws = None
                yield None
                await asyncio.sleep(backoff + random.uniform(0, backoff))
                backoff = min(backoff * 2, 30)

    async def close(self) -> None:
        """Close the underlying WebSocket connection."""
        if self._ws is not None:
            await self._ws.close()
            self._ws = None
