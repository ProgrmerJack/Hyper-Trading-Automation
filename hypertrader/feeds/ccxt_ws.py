from __future__ import annotations

from typing import AsyncIterator, Dict

import ccxt.async_support as ccxt


class CCXTWebSocketFeed:
    """Simple WebSocket market data feed using CCXT's built-in methods.

    Parameters
    ----------
    exchange : str
        Exchange identifier supported by ``ccxt`` (e.g. ``"binance"``).
    symbol : str
        Trading symbol in ``BASE/QUOTE`` format.
    """

    def __init__(self, exchange: str, symbol: str) -> None:
        self.exchange_name = exchange
        self.symbol = symbol
        self.client = getattr(ccxt, exchange)({"enableRateLimit": True})

    async def stream(self) -> AsyncIterator[Dict]:
        """Yield ticker updates indefinitely."""
        while True:
            tick = await self.client.watch_ticker(self.symbol)
            yield tick

    async def close(self) -> None:
        """Close the underlying WebSocket connection."""
        await self.client.close()
