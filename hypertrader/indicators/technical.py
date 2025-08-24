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

from typing import Iterable, Tuple
import numpy as np  # type: ignore

from hypertrader.utils.features import (
    ema as _ema,
    rsi as _rsi,
    macd as _macd,
    atr as _atr,
    # Import compute_bollinger_bands
    compute_bollinger_bands as _bollinger_bands,
    compute_supertrend as _supertrend,
    compute_anchored_vwap as _anchored_vwap,
    # Import correct function names
    compute_obv as _obv,
    compute_wavetrend as _wavetrend,
    compute_multi_rsi as _multi_rsi,
    # Import missing indicators
    compute_ichimoku as _ichimoku,
    compute_parabolic_sar as _parabolic_sar,
    compute_keltner_channels as _keltner_channels,
    compute_cci as _cci,
    compute_fibonacci_retracements as _fibonacci_retracements,
    compute_twap as _twap,
    compute_cumulative_delta as _cumulative_delta,
    compute_exchange_netflow as _exchange_netflow,
    compute_moving_average as _sma_func,
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

    This wrapper converts the input to a pandas Series and then
    delegates to the upstream ``bollinger_bands`` implementation.
    The upstream function returns a tuple of three Series; we
    extract the latest values and cast to floats.

    Parameters
    ----------
    data : Iterable[float]
        Price series (e.g., closing prices).
    period : int, optional
        Lookback window for the moving average and standard deviation.
    num_std : float, optional
        Number of standard deviations used for the upper and lower bands.

    Returns
    -------
    tuple
        The latest midpoint (SMA), upper band and lower band values.
    """
    try:
        # Try the list-based version first
        mid, upper, lower = _bollinger_bands(list(data), period, num_std)
        return float(mid[-1]), float(upper[-1]), float(lower[-1])
    except (TypeError, AttributeError):
        # Fall back to pandas-based version
        import pandas as pd
        series = pd.Series(list(data), dtype=float)
        mid, upper, lower = _bollinger_bands(series, period, num_std)
        return float(mid.iloc[-1]), float(upper.iloc[-1]), float(lower.iloc[-1])


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

    Constructs a DataFrame with ``close`` and ``volume`` columns and
    delegates to :func:`hypertrader.utils.features.compute_anchored_vwap`.
    The upstream function returns a pandas Series; we return the
    latest value.

    Parameters
    ----------
    close : iterable of float
        Closing price series.
    volume : iterable of float
        Volume series.

    Returns
    -------
    float
        The most recent anchored VWAP value.
    """
    try:
        # Try list-based version first
        return float(_anchored_vwap(list(close), list(volume))[-1])
    except (TypeError, AttributeError):
        # Fall back to DataFrame-based version
        import pandas as pd
        df = pd.DataFrame({"close": list(close), "volume": list(volume)}, dtype=float)
        vwap_series = _anchored_vwap(df)
        return float(vwap_series.iloc[-1])


def obv(close: Iterable[float], volume: Iterable[float]) -> float:
    """On balance volume.

    Wraps :func:`hypertrader.utils.features.obv` and returns the
    latest cumulative OBV value.  The upstream function expects a
    pandas DataFrame with ``'close'`` and ``'volume'`` columns.  This
    wrapper constructs such a DataFrame from the provided iterables.

    Parameters
    ----------
    close : iterable of float
        Sequence of closing prices.
    volume : iterable of float
        Sequence of traded volumes.

    Returns
    -------
    float
        The most recent OBV value.
    """
    try:
        # Try list-based version first
        return float(_obv(list(close), list(volume))[-1])
    except (TypeError, AttributeError):
        # Fall back to DataFrame-based version
        import pandas as pd
        df = pd.DataFrame({"close": list(close), "volume": list(volume)}, dtype=float)
        obv_series = _obv(df)
        return float(obv_series.iloc[-1])


def wavetrend(close: Iterable[float], period: int = 10) -> float:
    """WaveTrend oscillator wrapper.

    The upstream implementation requires a pandas DataFrame with
    ``'high'``, ``'low'`` and ``'close'`` columns.  This wrapper
    constructs such a DataFrame using the provided closing prices for
    all three columns (which approximates the typical price).  The
    resulting WaveTrend series is returned as a float for the most
    recent value.

    Parameters
    ----------
    close : iterable of float
        Closing prices.
    period : int, optional
        Lookback period for the oscillator.

    Returns
    -------
    float
        Latest WaveTrend oscillator value.
    """
    try:
        # Try list-based version first
        wt = _wavetrend(list(close), period)
        return float(wt[-1]) if isinstance(wt, (list, tuple, np.ndarray)) else float(wt)
    except (TypeError, AttributeError):
        # Fall back to DataFrame-based version
        import pandas as pd
        c = list(close)
        df = pd.DataFrame({"high": c, "low": c, "close": c}, dtype=float)
        wt_series = _wavetrend(df, period)
        if hasattr(wt_series, 'iloc'):
            return float(wt_series.iloc[-1])
        elif isinstance(wt_series, (list, tuple)):
            return float(wt_series[-1])
        else:
            return float(wt_series)


def multi_rsi(data: Iterable[float], periods: Iterable[int]) -> float:
    """Average of multiple RSI values over different periods.

    Wraps the upstream :func:`hypertrader.utils.features.multi_rsi`,
    which expects a pandas Series.  We convert the input list into a
    Series and return the most recent aggregated RSI value.

    Parameters
    ----------
    data : iterable of float
        Price series (e.g., close prices).
    periods : iterable of int
        RSI lookback periods to average.

    Returns
    -------
    float
        Latest averaged RSI.
    """
    try:
        # Try list-based version first
        mrs = _multi_rsi(list(data), list(periods))
        return float(mrs[-1])
    except (TypeError, AttributeError):
        # Fall back to pandas-based version
        import pandas as pd
        series = pd.Series(list(data), dtype=float)
        mrs_series = _multi_rsi(series, list(periods))
        return float(mrs_series.iloc[-1])


def ichimoku(high: Iterable[float], low: Iterable[float], close: Iterable[float]) -> dict[str, float]:
    """Compute Ichimoku Cloud components.
    
    Returns
    -------
    dict
        Dictionary with keys: tenkan, kijun, senkou_a, senkou_b, chikou
    """
    try:
        import pandas as pd
        df = pd.DataFrame({"high": list(high), "low": list(low), "close": list(close)})
        result = _ichimoku(df)
        return {
            "tenkan": float(result["tenkan"].iloc[-1]),
            "kijun": float(result["kijun"].iloc[-1]),
            "senkou_a": float(result["senkou_a"].iloc[-1]) if not pd.isna(result["senkou_a"].iloc[-1]) else 0.0,
            "senkou_b": float(result["senkou_b"].iloc[-1]) if not pd.isna(result["senkou_b"].iloc[-1]) else 0.0,
            "chikou": float(result["chikou"].iloc[-1]) if not pd.isna(result["chikou"].iloc[-1]) else 0.0,
        }
    except Exception as e:
        import logging
        logging.warning(f"Ichimoku calculation failed: {e}")
        return {"tenkan": 0.0, "kijun": 0.0, "senkou_a": 0.0, "senkou_b": 0.0, "chikou": 0.0}


def parabolic_sar(high: Iterable[float], low: Iterable[float], step: float = 0.02, max_step: float = 0.2) -> float:
    """Compute Parabolic SAR indicator.
    
    Returns
    -------
    float
        Latest Parabolic SAR value
    """
    try:
        import pandas as pd
        df = pd.DataFrame({"high": list(high), "low": list(low)})
        result = _parabolic_sar(df, step, max_step)
        return float(result.iloc[-1])
    except Exception:
        return 0.0


def keltner_channels(high: Iterable[float], low: Iterable[float], close: Iterable[float], 
                    ema_period: int = 20, atr_period: int = 10, multiplier: float = 2.0) -> dict[str, float]:
    """Compute Keltner Channels.
    
    Returns
    -------
    dict
        Dictionary with keys: ema, upper, lower
    """
    try:
        import pandas as pd
        df = pd.DataFrame({"high": list(high), "low": list(low), "close": list(close)})
        result = _keltner_channels(df, ema_period, atr_period, multiplier)
        return {
            "ema": float(result["ema"].iloc[-1]),
            "upper": float(result["upper"].iloc[-1]),
            "lower": float(result["lower"].iloc[-1]),
        }
    except Exception:
        return {"ema": 0.0, "upper": 0.0, "lower": 0.0}


def cci(high: Iterable[float], low: Iterable[float], close: Iterable[float], period: int = 20) -> float:
    """Compute Commodity Channel Index (CCI).
    
    Returns
    -------
    float
        Latest CCI value
    """
    try:
        import pandas as pd
        df = pd.DataFrame({"high": list(high), "low": list(low), "close": list(close)})
        result = _cci(df, period)
        return float(result.iloc[-1])
    except Exception:
        return 0.0


def fibonacci_retracements(high: Iterable[float], low: Iterable[float], window: int = 50) -> dict[str, float]:
    """Compute Fibonacci retracement levels.
    
    Returns
    -------
    dict
        Dictionary with Fibonacci levels
    """
    try:
        import pandas as pd
        df = pd.DataFrame({"high": list(high), "low": list(low)})
        result = _fibonacci_retracements(df, window)
        return {
            "level_0.236": float(result["level_0.236"].iloc[-1]),
            "level_0.382": float(result["level_0.382"].iloc[-1]),
            "level_0.5": float(result["level_0.5"].iloc[-1]),
            "level_0.618": float(result["level_0.618"].iloc[-1]),
            "level_0.786": float(result["level_0.786"].iloc[-1]),
        }
    except Exception:
        return {"level_0.236": 0.0, "level_0.382": 0.0, "level_0.5": 0.0, "level_0.618": 0.0, "level_0.786": 0.0}


def twap(close: Iterable[float]) -> float:
    """Compute Time Weighted Average Price (TWAP).
    
    Returns
    -------
    float
        Latest TWAP value
    """
    try:
        import pandas as pd
        df = pd.DataFrame({"close": list(close)})
        result = _twap(df)
        return float(result.iloc[-1])
    except Exception:
        return 0.0


def cumulative_delta(buy_vol: Iterable[float], sell_vol: Iterable[float]) -> float:
    """Compute cumulative volume delta.
    
    Returns
    -------
    float
        Latest cumulative delta value
    """
    try:
        import pandas as pd
        df = pd.DataFrame({"buy_vol": list(buy_vol), "sell_vol": list(sell_vol)})
        result = _cumulative_delta(df)
        return float(result.iloc[-1])
    except Exception:
        return 0.0


def exchange_netflow(inflows: Iterable[float], outflows: Iterable[float]) -> float:
    """Compute exchange net flow.
    
    Returns
    -------
    float
        Latest net flow value
    """
    try:
        import pandas as pd
        df = pd.DataFrame({"inflows": list(inflows), "outflows": list(outflows)})
        result = _exchange_netflow(df)
        return float(result.iloc[-1])
    except Exception:
        return 0.0


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
    "ichimoku",
    "parabolic_sar",
    "keltner_channels",
    "cci",
    "fibonacci_retracements",
    "twap",
    "cumulative_delta",
    "exchange_netflow",
]
