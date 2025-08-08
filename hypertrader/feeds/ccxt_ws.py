from __future__ import annotations

from typing import AsyncIterator, Dict

try:  # pragma: no cover - optional dependency
    import ccxt.pro as ccxtpro
except Exception:  # pragma: no cover - handled gracefully during tests
    ccxtpro = None


class CCXTWebSocketFeed:
    """Simple WebSocket market data feed using ``ccxt.pro``.

    Parameters
    ----------
    exchange : str
        Exchange identifier supported by ``ccxt.pro`` (e.g. ``"binance"``).
    symbol : str
        Trading symbol in ``BASE/QUOTE`` format.
    """

    def __init__(self, exchange: str, symbol: str) -> None:
        if ccxtpro is None:
            raise RuntimeError("ccxt.pro is required for WebSocket feed support")
        self.exchange_name = exchange
        self.symbol = symbol
        self.client = getattr(ccxtpro, exchange)()

    async def stream(self) -> AsyncIterator[Dict]:
        """Yield ticker updates indefinitely."""
        while True:
            tick = await self.client.watch_ticker(self.symbol)
            yield tick

    async def close(self) -> None:
        """Close the underlying WebSocket connection."""
        await self.client.close()
