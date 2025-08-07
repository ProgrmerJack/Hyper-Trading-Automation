"""Utilities for scanning and ranking assets by volatility.

This module provides helpers to identify the most volatile trading
pairs based on Average True Range (ATR).  It can be used by the bot to
focus on assets with the largest price swings.
"""
from __future__ import annotations

from typing import Callable, Sequence, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from .features import compute_atr
from ..data.fetch_data import fetch_yahoo_ohlcv

# Type alias for a callable that fetches OHLCV data given a symbol
DataFetcher = Callable[[str], pd.DataFrame]


def rank_symbols_by_volatility(
    symbols: Sequence[str],
    period: int = 14,
    data_fetcher: DataFetcher = fetch_yahoo_ohlcv,
    max_workers: int | None = None,
) -> List[str]:
    """Rank symbols by ATR-based volatility with concurrent fetching.

    Parameters
    ----------
    symbols:
        Iterable of ticker symbols to evaluate.
    period:
        ATR lookback period.
    data_fetcher:
        Function that returns OHLCV ``DataFrame`` for a symbol. By
        default the project ``fetch_yahoo_ohlcv`` loader is used. A
        custom fetcher can be supplied for testing.
    max_workers:
        Optional thread pool size for concurrent fetching. If ``None``
        the executor will choose a sensible default based on the number
        of CPU cores.

    Returns
    -------
    list[str]
        Symbols ordered from highest to lowest ATR value. Symbols that
        fail to fetch data are skipped. If none succeed an empty list is
        returned.
    """

    vol_map: dict[str, float] = {}

    def _worker(sym: str) -> tuple[str, float] | None:
        try:
            df = data_fetcher(sym)
            atr_series = compute_atr(df, period)
            atr = float(atr_series.iloc[-1])
            if not pd.isna(atr):
                return sym, atr
        except Exception:
            return None
        return None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_worker, sym): sym for sym in symbols}
        for fut in as_completed(futures):
            result = fut.result()
            if result:
                sym, atr = result
                vol_map[sym] = atr

    ranked = sorted(vol_map, key=vol_map.get, reverse=True)
    return ranked


__all__ = ["rank_symbols_by_volatility"]
