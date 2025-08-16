"""
Technical indicators for time series analysis.

This module exposes a suite of common technical analysis functions
implemented in :mod:`hypertrader.utils.features` for convenience.  The
functions here are thin wrappers around those implementations and
provide documentation adapted to the high‑frequency trading context.

The functions support vectorised NumPy arrays, pandas Series or
simple Python lists as inputs.  All return values are floats or
lists/arrays depending on the input type.

Available functions
-------------------

ema(data, period)
    Exponential moving average of a price series.
sma(data, period)
    Simple moving average of a price series.
rsi(data, period)
    Relative strength index, a momentum oscillator.
macd(data, fast, slow, signal)
    Moving average convergence divergence oscillator.
atr(high, low, close, period)
    Average true range, a volatility indicator.
bollinger_bands(data, period, num_std)
    Upper and lower Bollinger bands.
supertrend(high, low, close, period, multiplier)
    Trend indicator using ATR to compute bands.
anchored_vwap(close, volume)
    Anchored volume‑weighted average price.
obv(close, volume)
    On balance volume, a cumulative volume indicator.
wavetrend(close, period)
    WaveTrend oscillator, capturing cyclical momentum.
multi_rsi(data, periods)
    Compute RSI over multiple timeframes and average them.

See Also
--------
hypertrader.utils.features
    The module containing the underlying implementations.
"""

from __future__ import annotations

import numpy as np  # type: ignore

from hypertrader.utils.features import (
    ema as _ema,
    rsi as _rsi,
    macd as _macd,
    atr as _atr,
    compute_bollinger_bands as _bollinger_bands,
    compute_supertrend as _supertrend,
    compute_anchored_vwap as _anchored_vwap,
    on_balance_volume as _obv,
    compute_wavetrend as _wavetrend,
    compute_multi_rsi as _multi_rsi,
)


def sma(data: Iterable[float], period: int) -> float:
    """Compute the simple moving average of the last ``period`` values.

    Parameters
    ----------
    data : iterable of float
        Sequence of price values; typically closing prices.
    period : int
        Number of periods to average over.

    Returns
    -------
    float
        Arithmetic mean of the trailing ``period`` elements of
        ``data``.  If fewer than ``period`` values are available,
        returns the mean of all available values.
    """
    arr = np.asarray(list(data), dtype=float)
    if len(arr) < 1:
        return float('nan')
    if len(arr) < period:
        return float(np.mean(arr))
    return float(np.mean(arr[-period:]))


def ema(data: Iterable[float], period: int) -> float:
    """Exponential moving average wrapper.

    Delegates to :func:`hypertrader.utils.features.ema` and returns the
    latest EMA value.  See that function for details.
    """
    return float(_ema(list(data), period)[-1])


def rsi(data: Iterable[float], period: int) -> float:
    """Relative strength index.

    Wraps :func:`hypertrader.utils.features.rsi` and returns the most
    recent RSI value.
    """
    return float(_rsi(list(data), period)[-1])


def macd(data: Iterable[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float]:
    """Moving average convergence divergence (MACD).

    Returns the latest MACD line and signal line values.  See
    :func:`hypertrader.utils.features.macd` for full documentation.
    """
    macd_line, signal_line = _macd(list(data), fast, slow, signal)
    return float(macd_line[-1]), float(signal_line[-1])


def atr(high: Iterable[float], low: Iterable[float], close: Iterable[float], period: int) -> float:
    """Average true range.

    Computes the most recent ATR using the implementation from
    :mod:`hypertrader.utils.features`.  The three series must all be
    aligned and of equal length.  See that module for details on
    handling the first period.
    """
    return float(_atr(list(high), list(low), list(close), period)[-1])


def bollinger_bands(data: Iterable[float], period: int = 20, num_std: float = 2.0) -> Tuple[float, float, float]:
    """Compute Bollinger bands.

    Returns a tuple ``(middle, upper, lower)`` for the latest point.
    """
    mid, upper, lower = _bollinger_bands(list(data), period, num_std)
    return float(mid[-1]), float(upper[-1]), float(lower[-1])


def supertrend(high: Iterable[float], low: Iterable[float], close: Iterable[float], period: int = 10, multiplier: float = 3.0) -> float:
    """Compute SuperTrend indicator and return the current trend direction.

    The function returns a positive value if the trend is up and a
    negative value if the trend is down.  This is a simplified
    interface to :func:`hypertrader.utils.features.compute_supertrend`.
    """
    st, direction = _supertrend(list(high), list(low), list(close), period, multiplier)
    return float(direction[-1])


def anchored_vwap(close: Iterable[float], volume: Iterable[float]) -> float:
    """Anchored VWAP wrapper.

    Returns the latest volume‑weighted average price anchored at the
    beginning of the provided series.
    """
    return float(_anchored_vwap(list(close), list(volume))[-1])


def obv(close: Iterable[float], volume: Iterable[float]) -> float:
    """On balance volume.

    Wraps :func:`hypertrader.utils.features.on_balance_volume` and
    returns the latest cumulative OBV value.
    """
    return float(_obv(list(close), list(volume))[-1])


def wavetrend(close: Iterable[float], period: int = 10) -> float:
    """WaveTrend oscillator wrapper.

    Returns the latest WaveTrend oscillator value.
    """
    wt = _wavetrend(list(close), period)
    return float(wt[-1]) if isinstance(wt, (list, tuple, np.ndarray)) else float(wt)


def multi_rsi(data: Iterable[float], periods: Iterable[int]) -> float:
    """Average of multiple RSI values over different periods.

    Calls :func:`hypertrader.utils.features.compute_multi_rsi` and
    returns the most recent aggregated RSI value.  Useful for
    smoothing noise across timeframes.
    """
    mrs = _multi_rsi(list(data), list(periods))
    return float(mrs[-1])


__all__ = [
    "sma",
    "ema",
    "rsi",
    "macd",
    "atr",
    "bollinger_bands",
    "supertrend",
    "anchored_vwap",
    "obv",
    "wavetrend",
    "multi_rsi",
]
