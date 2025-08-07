import pandas as pd
import numpy as np


def onchain_zscore(df: pd.DataFrame, window: int = 30) -> pd.Series:
    """Compute z-score of on-chain metrics such as gas fees.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a ``gas`` column.
    window : int, optional
        Rolling window used for mean and standard deviation, by default 30.

    Returns
    -------
    pd.Series
        Z-score series where values above 1.5 indicate elevated activity.
    """
    if "gas" not in df:
        raise ValueError("DataFrame must contain 'gas' column")
    mean = df["gas"].rolling(window).mean()
    std = df["gas"].rolling(window).std()
    z = (df["gas"] - mean) / std
    return z.fillna(0)


def order_skew(order_book: dict, depth: int = 5) -> float:
    """Compute order book imbalance between bid and ask volumes.

    Parameters
    ----------
    order_book : dict
        Mapping containing ``bids`` and ``asks`` lists as returned by CCXT.
    depth : int, optional
        Number of levels to aggregate, by default 5.

    Returns
    -------
    float
        Value in range [-1, 1] where positive indicates bid dominance.
    """
    bids = order_book.get("bids", [])[:depth]
    asks = order_book.get("asks", [])[:depth]
    bid_vol = sum(level[1] for level in bids)
    ask_vol = sum(level[1] for level in asks)
    total = bid_vol + ask_vol
    if total == 0:
        return 0.0
    return (bid_vol - ask_vol) / total


def dom_heatmap_ratio(order_book: dict, layers: int = 10) -> float:
    """Compute bid/ask volume ratio from the depth of market.

    Aggregates volume across the top ``layers`` of the order book and
    returns the ratio of bid volume to ask volume. Values above ``1``
    indicate bid dominance while values below ``1`` indicate ask
    dominance.

    Parameters
    ----------
    order_book : dict
        Mapping containing ``bids`` and ``asks`` lists as returned by CCXT.
    layers : int, optional
        Number of price levels to aggregate, by default ``10``.

    Returns
    -------
    float
        Bid to ask volume ratio. If ask volume is zero the function
        returns ``float('inf')`` when bids are present or ``1.0`` when the
        book is empty.
    """
    bids = order_book.get("bids", [])[:layers]
    asks = order_book.get("asks", [])[:layers]
    bid_vol = sum(level[1] for level in bids)
    ask_vol = sum(level[1] for level in asks)
    if ask_vol == 0:
        return float("inf") if bid_vol > 0 else 1.0
    return bid_vol / ask_vol


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """Compute the Volume Weighted Average Price (VWAP).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ``close`` and ``volume`` columns.

    Returns
    -------
    pd.Series
        Series of cumulative VWAP values.
    """
    if not {"close", "volume"}.issubset(df.columns):
        raise ValueError("DataFrame must contain close and volume columns")
    pv = (df["close"] * df["volume"]).cumsum()
    vol = df["volume"].cumsum()
    return pv / vol


def compute_obv(df: pd.DataFrame) -> pd.Series:
    """Compute On-Balance Volume indicator."""
    if not {"close", "volume"}.issubset(df.columns):
        raise ValueError("DataFrame must contain close and volume columns")
    obv = pd.Series(index=df.index, dtype=float)
    obv.iloc[0] = 0
    for i in range(1, len(df)):
        if df["close"].iloc[i] > df["close"].iloc[i - 1]:
            obv.iloc[i] = obv.iloc[i - 1] + df["volume"].iloc[i]
        elif df["close"].iloc[i] < df["close"].iloc[i - 1]:
            obv.iloc[i] = obv.iloc[i - 1] - df["volume"].iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i - 1]
    return obv


def compute_moving_average(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window).mean()


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average used by many TradingView strategies."""
    return series.ewm(span=span, adjust=False).mean()


def compute_rsi(price_series: pd.Series, period: int = 14) -> pd.Series:
    delta = price_series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(price_series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = price_series.ewm(span=fast, adjust=False).mean()
    ema_slow = price_series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - macd_signal
    return pd.DataFrame({'macd': macd, 'signal': macd_signal, 'histogram': histogram})


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range indicator."""
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr


def compute_bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Compute Bollinger Bands."""
    ma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    return pd.DataFrame({'ma': ma, 'upper': upper, 'lower': lower})


def compute_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """Compute SuperTrend indicator widely shared on TradingView."""
    atr = compute_atr(df, period)
    hl2 = (df['high'] + df['low']) / 2
    upperband = hl2 + multiplier * atr
    lowerband = hl2 - multiplier * atr

    final_upper = upperband.copy()
    final_lower = lowerband.copy()
    for i in range(1, len(df)):
        final_upper.iloc[i] = (
            upperband.iloc[i]
            if df['close'].iloc[i - 1] > final_upper.iloc[i - 1]
            else min(upperband.iloc[i], final_upper.iloc[i - 1])
        )
        final_lower.iloc[i] = (
            lowerband.iloc[i]
            if df['close'].iloc[i - 1] < final_lower.iloc[i - 1]
            else max(lowerband.iloc[i], final_lower.iloc[i - 1])
        )

    direction = pd.Series(index=df.index, dtype=int)
    supertrend = pd.Series(index=df.index, dtype=float)
    direction.iloc[0] = -1
    supertrend.iloc[0] = upperband.iloc[0]
    for i in range(1, len(df)):
        if supertrend.iloc[i - 1] == final_upper.iloc[i - 1]:
            supertrend.iloc[i] = (
                final_upper.iloc[i]
                if df['close'].iloc[i] <= final_upper.iloc[i]
                else final_lower.iloc[i]
            )
        else:
            supertrend.iloc[i] = (
                final_lower.iloc[i]
                if df['close'].iloc[i] >= final_lower.iloc[i]
                else final_upper.iloc[i]
            )
        direction.iloc[i] = 1 if supertrend.iloc[i] < df['close'].iloc[i] else -1

    return pd.DataFrame({'supertrend': supertrend, 'direction': direction})


def compute_anchored_vwap(df: pd.DataFrame, anchor: str = 'high') -> pd.Series:
    """Compute Anchored VWAP from the last significant high or low.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least ``high``, ``low``, ``close`` and ``volume`` columns.
    anchor : str, optional
        Anchor point: ``'high'`` uses the location of the maximum high, ``'low'`` uses the
        location of the minimum low. Default is ``'high'``.

    Returns
    -------
    pd.Series
        Series of anchored VWAP values with ``NaN`` before the anchor index.
    """
    if anchor not in {'high', 'low'}:
        raise ValueError("anchor must be 'high' or 'low'")
    if not {'high', 'low', 'close', 'volume'}.issubset(df.columns):
        raise ValueError("DataFrame must contain high, low, close, volume columns")

    anchor_idx = df['high'].idxmax() if anchor == 'high' else df['low'].idxmin()
    subset = df.loc[anchor_idx:]
    pv = (subset['close'] * subset['volume']).cumsum()
    v = subset['volume'].cumsum()
    vwap = pv / v
    anchored = pd.Series(index=df.index, dtype=float)
    anchored.loc[anchor_idx:] = vwap
    return anchored


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average Directional Index (ADX)."""
    if not {"high", "low", "close"}.issubset(df.columns):
        raise ValueError("DataFrame must contain high, low and close columns")

    up_move = df["high"].diff()
    down_move = df["low"].shift() - df["low"]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.rolling(window=period).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(window=period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=period).mean() / atr
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100
    adx = dx.rolling(window=period).mean()
    return adx


def compute_stochastic(
    df: pd.DataFrame, k_period: int = 14, d_period: int = 3
) -> pd.DataFrame:
    """Compute Stochastic Oscillator values."""
    if not {"high", "low", "close"}.issubset(df.columns):
        raise ValueError("DataFrame must contain high, low and close columns")
    low_min = df["low"].rolling(window=k_period).min()
    high_max = df["high"].rolling(window=k_period).max()
    k = 100 * (df["close"] - low_min) / (high_max - low_min)
    d = k.rolling(window=d_period).mean()
    return pd.DataFrame({"k": k, "d": d})


def compute_roc(series: pd.Series, period: int = 5) -> pd.Series:
    """Compute Rate of Change indicator."""
    return series.pct_change(periods=period) * 100


def compute_twap(df: pd.DataFrame) -> pd.Series:
    """Compute Time Weighted Average Price (TWAP)."""
    if "close" not in df:
        raise ValueError("DataFrame must contain close column")
    return df["close"].expanding().mean()


def compute_cumulative_delta(df: pd.DataFrame) -> pd.Series:
    """Compute cumulative volume delta from buy and sell volumes."""
    if not {"buy_vol", "sell_vol"}.issubset(df.columns):
        raise ValueError("DataFrame must contain buy_vol and sell_vol columns")
    delta = df["buy_vol"] - df["sell_vol"]
    return delta.cumsum()


def compute_cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Compute Commodity Channel Index (CCI)."""
    if not {"high", "low", "close"}.issubset(df.columns):
        raise ValueError("DataFrame must contain high, low and close columns")
    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
    cci = (tp - sma) / (0.015 * mad)
    return cci


def compute_keltner_channels(
    df: pd.DataFrame, ema_period: int = 20, atr_period: int = 10, multiplier: float = 2.0
) -> pd.DataFrame:
    """Compute Keltner Channels for a price series."""
    if not {"high", "low", "close"}.issubset(df.columns):
        raise ValueError("DataFrame must contain high, low and close columns")
    ema = df["close"].ewm(span=ema_period, adjust=False).mean()
    atr = compute_atr(df, period=atr_period)
    upper = ema + multiplier * atr
    lower = ema - multiplier * atr
    return pd.DataFrame({"ema": ema, "upper": upper, "lower": lower})


def compute_exchange_netflow(df: pd.DataFrame) -> pd.Series:
    """Compute net flow from exchange inflows and outflows.

    The function expects ``inflows`` and ``outflows`` columns and returns the
    difference ``outflows - inflows`` which can be used to gauge supply and
    demand pressure on an exchange.
    """
    if not {"inflows", "outflows"}.issubset(df.columns):
        raise ValueError("DataFrame must contain inflows and outflows columns")
    return df["outflows"] - df["inflows"]


def compute_volatility_cluster(df: pd.DataFrame, window: int = 10) -> pd.Series:
    """Return a simple volatility clustering index.

    Calculates rolling variance of returns and marks periods where variance is
    more than one standard deviation above its rolling mean. The resulting
    series is the density of such high-volatility observations within the
    specified window and ranges between 0 and 1.
    """
    if "close" not in df:
        raise ValueError("DataFrame must contain close column")
    returns = df["close"].pct_change()
    var = returns.rolling(window).var()
    mean_var = var.rolling(window).mean()
    std_var = var.rolling(window).std()
    spikes = (var > (mean_var + std_var)).astype(float)
    cluster = spikes.rolling(window).mean()
    return cluster.fillna(0)


def compute_fibonacci_retracements(
    df: pd.DataFrame, window: int = 50
) -> pd.DataFrame:
    """Compute Fibonacci retracement levels for a rolling window.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``high`` and ``low`` columns.
    window : int, optional
        Rolling window to determine swing high and low, by default ``50``.

    Returns
    -------
    pd.DataFrame
        Columns for each common Fibonacci level from 23.6% to 78.6%.
    """
    if not {"high", "low"}.issubset(df.columns):
        raise ValueError("DataFrame must contain high and low columns")
    swing_high = df["high"].rolling(window).max()
    swing_low = df["low"].rolling(window).min()
    diff = swing_high - swing_low
    levels = {
        "level_0.236": swing_low + diff * 0.236,
        "level_0.382": swing_low + diff * 0.382,
        "level_0.5": swing_low + diff * 0.5,
        "level_0.618": swing_low + diff * 0.618,
        "level_0.786": swing_low + diff * 0.786,
    }
    return pd.DataFrame(levels)


def compute_ai_momentum(series: pd.Series, period: int = 10) -> pd.Series:
    """Estimate momentum using a simple linear regression slope.

    This lightweight proxy for a learning-based indicator fits a first-degree
    polynomial over the last ``period`` closing prices and returns the slope as
    a momentum estimate.
    """
    if len(series) < period:
        raise ValueError("Series must be at least as long as the period")

    def _slope(window_values: pd.Series) -> float:
        y = window_values.values
        x = np.arange(len(y))
        return float(np.polyfit(x, y, 1)[0])

    return series.rolling(window=period).apply(_slope, raw=False)


def compute_wavetrend(
    df: pd.DataFrame, channel_len: int = 10, avg_len: int = 21
) -> pd.Series:
    """Compute WaveTrend oscillator used in many TradingView scripts.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ``high``, ``low`` and ``close`` columns.
    channel_len : int, optional
        Length of the EMA channel, by default ``10``.
    avg_len : int, optional
        Length of the signal smoothing average, by default ``21``.

    Returns
    -------
    pd.Series
        WaveTrend oscillator values.
    """
    if not {"high", "low", "close"}.issubset(df.columns):
        raise ValueError("DataFrame must contain high, low and close columns")
    ap = (df["high"] + df["low"] + df["close"]) / 3
    esa = ap.ewm(span=channel_len, adjust=False).mean()
    d = (ap - esa).abs().ewm(span=channel_len, adjust=False).mean()
    ci = (ap - esa) / (0.015 * d)
    wt = ci.ewm(span=avg_len, adjust=False).mean()
    return wt


def compute_multi_rsi(
    df: pd.DataFrame, periods: list[int] | None = None, weights: list[float] | None = None
) -> pd.Series:
    """Compute weighted RSI across multiple timeframes.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a datetime index and a ``close`` column.
    periods : list[int], optional
        Timeframe lengths in minutes, by default ``[5, 15, 60]``.
    weights : list[float], optional
        Weighting for each timeframe, defaults to equal weighting.

    Returns
    -------
    pd.Series
        Weighted RSI series aligned to the original index.
    """
    if "close" not in df:
        raise ValueError("DataFrame must contain close column")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")
    periods = periods or [5, 15, 60]
    weights = weights or [1 / len(periods)] * len(periods)
    if len(periods) != len(weights):
        raise ValueError("periods and weights must be the same length")

    rsi_total = pd.Series(0.0, index=df.index)
    for p, w in zip(periods, weights):
        resampled = df["close"].resample(f"{p}min").last()
        if len(resampled) < 2:
            rsi = pd.Series(50.0, index=resampled.index)
        else:
            rsi_period = min(14, len(resampled) - 1)
            rsi = compute_rsi(resampled, period=rsi_period)
        rsi_total += rsi.reindex(df.index, method="ffill") * w
    return rsi_total


def compute_vpvr_poc(df: pd.DataFrame, bins: int = 50) -> float:
    """Return the volume point of control from Volume Profile Visible Range."""
    if not {"close", "volume"}.issubset(df.columns):
        raise ValueError("DataFrame must contain close and volume columns")
    hist, edges = np.histogram(df["close"], bins=bins, weights=df["volume"])
    idx = int(hist.argmax())
    return float((edges[idx] + edges[idx + 1]) / 2)


def compute_ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    """Compute basic Ichimoku Cloud components."""
    if not {"high", "low", "close"}.issubset(df.columns):
        raise ValueError("DataFrame must contain high, low and close columns")
    tenkan = (df["high"].rolling(9).max() + df["low"].rolling(9).min()) / 2
    kijun = (df["high"].rolling(26).max() + df["low"].rolling(26).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = ((df["high"].rolling(52).max() + df["low"].rolling(52).min()) / 2).shift(26)
    chikou = df["close"].shift(-26)
    return pd.DataFrame(
        {
            "tenkan": tenkan,
            "kijun": kijun,
            "senkou_a": senkou_a,
            "senkou_b": senkou_b,
            "chikou": chikou,
        }
    )


def compute_parabolic_sar(
    df: pd.DataFrame, step: float = 0.02, max_step: float = 0.2
) -> pd.Series:
    """Compute Parabolic SAR indicator."""
    if not {"high", "low"}.issubset(df.columns):
        raise ValueError("DataFrame must contain high and low columns")
    high = df["high"].values
    low = df["low"].values
    sar = np.zeros(len(df))
    uptrend = True
    ep = high[0]
    af = step
    sar[0] = low[0]
    for i in range(1, len(df)):
        prev_sar = sar[i - 1]
        sar[i] = prev_sar + af * (ep - prev_sar)
        if uptrend:
            sar[i] = min(sar[i], low[i - 1], low[i])
            if high[i] > ep:
                ep = high[i]
                af = min(af + step, max_step)
            if low[i] < sar[i]:
                uptrend = False
                sar[i] = ep
                ep = low[i]
                af = step
        else:
            sar[i] = max(sar[i], high[i - 1], high[i])
            if low[i] < ep:
                ep = low[i]
                af = min(af + step, max_step)
            if high[i] > sar[i]:
                uptrend = True
                sar[i] = ep
                ep = high[i]
                af = step
    return pd.Series(sar, index=df.index)
