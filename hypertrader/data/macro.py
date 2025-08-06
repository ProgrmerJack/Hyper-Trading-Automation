import pandas as pd
import yfinance as yf
from fredapi import Fred

from ..utils.net import fetch_with_retry


def fetch_dxy(interval: str = '1d', lookback: str = '6mo', api_key: str | None = None) -> pd.Series:
    """Fetch the US Dollar Index (DXY) from Yahoo Finance with optional FRED fallback."""

    def _download():
        return yf.download('DX-Y.NYB', period=lookback, interval=interval, progress=False)

    try:
        df = fetch_with_retry(_download)
        if df.empty:
            raise ValueError('No data fetched for DXY')
        df.index.name = 'timestamp'
        return df['Close']
    except Exception:
        if api_key:
            # Fallback to trade weighted dollar index from FRED
            series = fetch_fred_series('DTWEXBGS', api_key)
            return series
        raise


def fetch_fred_series(series_id: str, api_key: str) -> pd.Series:
    """Fetch a data series from the FRED API with retry logic."""

    fred = Fred(api_key=api_key)

    def _get():
        return fred.get_series(series_id)

    series = fetch_with_retry(_get)
    if series is None or series.empty:
        raise ValueError(f'No data for {series_id}')
    series.index.name = 'timestamp'
    return series


def fetch_interest_rate(api_key: str) -> pd.Series:
    """Fetch effective federal funds rate."""
    return fetch_fred_series('FEDFUNDS', api_key)


def fetch_global_liquidity(api_key: str) -> pd.Series:
    """Fetch global liquidity proxy (M2 money stock)."""
    return fetch_fred_series('M2SL', api_key)


def fetch_cardboard_production(api_key: str) -> pd.Series:
    """Fetch cardboard production index used as an unconventional leading indicator."""
    return fetch_fred_series('IPN32221S', api_key)
