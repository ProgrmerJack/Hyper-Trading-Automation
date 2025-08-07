import ccxt
import pandas as pd
import yfinance as yf
from datetime import datetime

from tenacity import retry, stop_after_attempt, wait_exponential

from ..utils.net import fetch_with_retry


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def _ccxt_fetch(exchange_name: str, symbol: str, timeframe: str, since: int | None, limit: int) -> list[list[float]]:
    exchange_class = getattr(ccxt, exchange_name)
    exchange = exchange_class()
    return exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)


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


def fetch_yahoo_ohlcv(symbol: str, interval: str = '1h', lookback: str = '7d') -> pd.DataFrame:
    """Fetch OHLCV data from Yahoo Finance with retry logic."""

    def _download():
        return yf.download(symbol, period=lookback, interval=interval, progress=False)

    df = fetch_with_retry(_download)
    if df.empty:
        raise ValueError('No data fetched from Yahoo Finance')

    # yfinance may return a MultiIndex with the ticker as the second level
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index.name = 'timestamp'
    df.rename(columns={c: c.lower() for c in df.columns}, inplace=True)
    return df[['open', 'high', 'low', 'close', 'volume']]


def fetch_order_book(exchange_name: str, symbol: str, limit: int = 5) -> dict:
    """Fetch order book data from an exchange using CCXT."""
    exchange_class = getattr(ccxt, exchange_name)
    exchange = exchange_class()
    return exchange.fetch_order_book(symbol, limit=limit)
