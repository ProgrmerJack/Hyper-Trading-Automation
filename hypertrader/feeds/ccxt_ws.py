from __future__ import annotations

from typing import AsyncIterator, Dict

import asyncio
import time

import ccxt.async_support as ccxt


class CCXTWebSocketFeed:
    """Robust WebSocket market data feed using CCXT's built-in methods.

    The feed automatically reconnects on network errors and enforces a
    heartbeat so stale connections are detected and refreshed.  Each
    reconnect uses an exponential backoff capped at 30 seconds.

    Parameters
    ----------
    exchange : str
        Exchange identifier supported by ``ccxt`` (e.g. ``"binance"``).
    symbol : str
        Trading symbol in ``BASE/QUOTE`` format.
    heartbeat : int, optional
        Maximum seconds to wait for a message before reconnecting.  Defaults
        to 30 seconds.
    """

    def __init__(self, exchange: str, symbol: str, heartbeat: int = 30) -> None:
        self.exchange_name = exchange
        self.symbol = symbol
        self.heartbeat = heartbeat
        self.client = getattr(ccxt, exchange)({"enableRateLimit": True})
        self._last_msg = time.time()

    async def _reconnect(self) -> None:
        await self.client.close()
        self.client = getattr(ccxt, self.exchange_name)({"enableRateLimit": True})

    async def stream(self) -> AsyncIterator[Dict]:
        """Yield ticker updates indefinitely with automatic reconnection."""
        backoff = 1
        while True:
            try:
                tick = await asyncio.wait_for(
                    self.client.watch_ticker(self.symbol), timeout=self.heartbeat
                )
                self._last_msg = time.time()
                backoff = 1
                yield tick
            except (asyncio.TimeoutError, ccxt.BaseError):
                await self._reconnect()
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)

    async def close(self) -> None:
        """Close the underlying WebSocket connection."""
        await self.client.close()
