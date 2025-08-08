import pandas as pd
from fredapi import Fred

from ..utils.net import fetch_with_retry


def fetch_dxy(api_key: str) -> pd.Series:
    """Fetch the trade-weighted dollar index from FRED."""
    return fetch_fred_series("DTWEXBGS", api_key)


def fetch_fred_series(series_id: str, api_key: str) -> pd.Series:
    """Fetch a data series from the FRED API with retry logic."""
    fred = Fred(api_key=api_key)

    def _get():
        return fred.get_series(series_id)

    series = fetch_with_retry(_get)
    if series is None or series.empty:
        raise ValueError(f"No data for {series_id}")
    series.index.name = "timestamp"
    return series


def fetch_interest_rate(api_key: str) -> pd.Series:
    """Fetch effective federal funds rate."""
    return fetch_fred_series("FEDFUNDS", api_key)


def fetch_global_liquidity(api_key: str) -> pd.Series:
    """Fetch global liquidity proxy (M2 money stock)."""
    return fetch_fred_series("M2SL", api_key)


def fetch_cardboard_production(api_key: str) -> pd.Series:
    """Fetch cardboard production index used as an unconventional leading indicator."""
    return fetch_fred_series("IPN32221S", api_key)
