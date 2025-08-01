import pandas as pd
import yfinance as yf
from fredapi import Fred


def fetch_dxy(interval: str = '1d', lookback: str = '6mo') -> pd.Series:
    """Fetch the US Dollar Index (DXY) from Yahoo Finance."""
    df = yf.download('DX-Y.NYB', period=lookback, interval=interval, progress=False)
    if df.empty:
        raise ValueError('No data fetched for DXY')
    df.index.name = 'timestamp'
    return df['Close']


def fetch_fred_series(series_id: str, api_key: str) -> pd.Series:
    """Fetch a data series from the FRED API."""
    fred = Fred(api_key=api_key)
    series = fred.get_series(series_id)
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
