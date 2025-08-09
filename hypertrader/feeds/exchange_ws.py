from __future__ import annotations

import asyncio
import json
import time
from typing import AsyncIterator, Dict, Optional

import websockets


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
        """Yield ticker messages indefinitely with automatic reconnection."""
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
                msg = await asyncio.wait_for(self._ws.recv(), timeout=self.heartbeat)
                self._last_msg = time.time()
                yield json.loads(msg)
            except Exception:
                try:
                    if self._ws is not None:
                        await self._ws.close()
                finally:
                    self._ws = None
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)

    async def close(self) -> None:
        """Close the underlying WebSocket connection."""
        if self._ws is not None:
            await self._ws.close()
            self._ws = None
