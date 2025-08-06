import pandas as pd


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
