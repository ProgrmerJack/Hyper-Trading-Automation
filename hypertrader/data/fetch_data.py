import ccxt
import asyncio
import pandas as pd
from importlib import import_module
from typing import AsyncIterator, Dict, Any, Optional

from tenacity import retry, stop_after_attempt, wait_exponential


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


async def websocket_ingest(symbol: str, exchange_name: str = "binance") -> AsyncIterator[Dict[str, Any]]:
    """Stream ticker updates via WebSocket using ``ccxt.pro``.

    Parameters
    ----------
    symbol : str
        Trading pair symbol like ``'BTC/USDT'``.
    exchange_name : str, default ``"binance"``
        Exchange name supported by ``ccxt.pro``.

    Yields
    ------
    dict
        Raw ticker information from the exchange.

    Notes
    -----
    Requires the optional ``ccxt.pro`` package. A minimal example:

    >>> async for tick in websocket_ingest('BTC/USDT'):
    ...     print(tick['last'])
    """

    try:  # pragma: no cover - optional dependency
        ccxtpro = import_module("ccxt.pro")
    except Exception as exc:  # pragma: no cover - network/optional
        raise ImportError("ccxt.pro is required for websocket_ingest") from exc

    exchange_class = getattr(ccxtpro, exchange_name)
    exchange = exchange_class()
    while True:
        ticker = await exchange.watch_ticker(symbol)
        yield ticker


async def stream_ohlcv(
    symbol: str,
    timeframe: str = "1m",
    exchange_name: str = "binance",
    queue: Optional[asyncio.Queue] = None,
) -> AsyncIterator[list[float]]:
    """Stream OHLCV candles via ``ccxt.pro``.

    Parameters
    ----------
    symbol : str
        Trading pair symbol like ``'BTC/USDT'``.
    timeframe : str, default ``"1m"``
        Candle timeframe to subscribe to.
    exchange_name : str, default ``"binance"``
        Exchange name supported by ``ccxt.pro``.
    queue : asyncio.Queue, optional
        If provided, received candles are placed into the queue instead of
        being yielded.
    """

    try:  # pragma: no cover - optional dependency
        ccxtpro = import_module("ccxt.pro")
    except Exception as exc:  # pragma: no cover - network/optional
        raise ImportError("ccxt.pro is required for stream_ohlcv") from exc

    exchange_class = getattr(ccxtpro, exchange_name)
    exchange = exchange_class()
    while True:
        candle = await exchange.watch_ohlcv(symbol, timeframe)
        if queue is not None:
            await queue.put(candle)
        else:
            yield candle
