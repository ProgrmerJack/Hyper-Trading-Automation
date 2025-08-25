import ccxt
import asyncio
import pandas as pd
from typing import AsyncIterator, Dict, Any, Optional
import time

from tenacity import retry, stop_after_attempt, wait_exponential

from ..feeds.exchange_ws import ExchangeWebSocketFeed


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def _ccxt_fetch(exchange_name: str, symbol: str, timeframe: str, since: int | None, limit: int) -> list[list[float]]:
    exchange_class = getattr(ccxt, exchange_name)
    exchange = exchange_class()
    try:
        return exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
    finally:
        # Ensure network resources are released even on failure
        exchange.close()


def fetch_ohlcv(
    exchange_name: str,
    symbol: str,
    timeframe: str = "1h",
    since: int | None = None,
    limit: int = 1000,
    fallback: str | None = None,
) -> pd.DataFrame:
    """Fetch OHLCV data from the given exchange using CCXT with retry and fallback.

    Parameters
    ----------
    exchange_name : str
        Name of the exchange supported by CCXT (e.g. 'binance').
    symbol : str
        Trading pair symbol like 'BTC/USDT'.
    timeframe : str
        Timeframe string such as '1h' or '1d'.
    since : int | None
        Timestamp in milliseconds to start fetching from.
    limit : int
        Number of candles to retrieve.

    fallback : str | None
        Optional fallback exchange name to use if the primary fails.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by datetime with columns [open, high, low, close, volume].
    """
    try:
        ohlcv = _ccxt_fetch(exchange_name, symbol, timeframe, since, limit)
    except Exception:
        if not fallback:
            raise
        ohlcv = _ccxt_fetch(fallback, symbol, timeframe, since, limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df




def fetch_order_book(exchange_name: str, symbol: str, limit: int = 5) -> dict:
    """Fetch order book data from an exchange using CCXT."""
    exchange_class = getattr(ccxt, exchange_name)
    exchange = exchange_class()
    try:
        return exchange.fetch_order_book(symbol, limit=limit)
    finally:
        exchange.close()


async def websocket_ingest(symbol: str, exchange_name: str = "binance", heartbeat: int = 30) -> AsyncIterator[Dict[str, Any]]:
    """Stream ticker updates via direct exchange WebSockets.

    Parameters
    ----------
    symbol : str
        Trading pair symbol like ``'BTC/USDT'``.
    exchange_name : str, default ``"binance"``
        Exchange identifier.
    heartbeat : int, optional
        Seconds to wait before considering the connection stale.

    Yields
    ------
    dict
        Raw ticker information from the exchange.
    """

    ws_symbol = symbol
    if exchange_name.lower() == "binance":
        ws_symbol = symbol.replace("/", "").replace("-", "").lower()
    elif exchange_name.lower() == "bybit":
        ws_symbol = symbol.replace("/", "").replace("-", "").upper()

    feed = ExchangeWebSocketFeed(exchange_name, ws_symbol, heartbeat=heartbeat)
    async for msg in feed.stream():
        if msg is None:
            continue
        yield msg


async def stream_ohlcv(
    symbol: str,
    timeframe: str = "1m",
    exchange_name: str = "binance",
    queue: Optional[asyncio.Queue] = None,
) -> AsyncIterator[list[float]]:
    """Stream OHLCV candles using ticker WebSocket data.

    This derives candle information from the live ticker feed as a lightweight
    alternative to CCXT Pro.  It currently supports Binance and Bybit tickers.
    """

    ws_symbol = symbol
    if exchange_name.lower() == "binance":
        ws_symbol = symbol.replace("/", "").replace("-", "").lower()
    elif exchange_name.lower() == "bybit":
        ws_symbol = symbol.replace("/", "").replace("-", "").upper()

    feed = ExchangeWebSocketFeed(exchange_name, ws_symbol)
    async for msg in feed.stream():
        if msg is None or not msg or not isinstance(msg, dict):
            continue
        ts = int(time.time() * 1000)
        if exchange_name.lower() == "binance":
            # Skip messages that don't have required fields
            if not all(key in msg for key in ["o", "h", "l", "c"]):
                continue
            candle = [
                ts,
                float(msg["o"]),
                float(msg["h"]),
                float(msg["l"]),
                float(msg["c"]),
                float(msg.get("v", 0.0)),
            ]
        else:  # bybit
            candle = [
                ts,
                float(msg["openPrice24h"]),
                float(msg["highPrice24h"]),
                float(msg["lowPrice24h"]),
                float(msg["lastPrice"]),
                float(msg.get("turnover24h", 0.0)),
            ]
        if queue is not None:
            await queue.put(candle)
        else:
            yield candle
