from dataclasses import dataclass
from typing import Optional
import pandas as pd

from ..utils.features import (
    compute_ema,
    compute_rsi,
    compute_bollinger_bands,
    compute_supertrend,
    compute_anchored_vwap,
    compute_vwap,
    compute_obv,
    compute_wavetrend,
    compute_multi_rsi,
)


@dataclass
class Signal:
    action: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    volume: float = 0.0


def generate_signal(
    data: pd.DataFrame,
    sentiment_score: float = 0.0,
    macro_score: float = 0.0,
    onchain_score: float = 0.0,
    book_skew: float = 0.0,
    heatmap_ratio: float = 1.0,
) -> Signal:
    """Generate trading signal from OHLCV dataframe.

    Uses moving average crossover and RSI rules combined with sentiment.
    """
    if len(data) < 200:
        return Signal('HOLD')
    short_ma = compute_ema(data['close'], 50).iloc[-1]
    long_ma = compute_ema(data['close'], 200).iloc[-1]
    rsi = compute_rsi(data['close']).iloc[-1]
    bands = compute_bollinger_bands(data['close'])
    upper = bands['upper'].iloc[-1]
    lower = bands['lower'].iloc[-1]

    if {'high', 'low'}.issubset(data.columns):
        st = compute_supertrend(data)
        direction = st['direction'].iloc[-1]
    else:
        direction = 0

    if {'high', 'low', 'volume'}.issubset(data.columns):
        anchor_high = compute_anchored_vwap(data, anchor='high').iloc[-1]
        anchor_low = compute_anchored_vwap(data, anchor='low').iloc[-1]
        vwap = compute_vwap(data).iloc[-1]
        obv = compute_obv(data).iloc[-1]
    else:
        anchor_high = anchor_low = float('nan')
        vwap = obv = float('nan')

    if {'high', 'low', 'close'}.issubset(data.columns):
        wt = compute_wavetrend(data).iloc[-1]
    else:
        wt = float('nan')

    if isinstance(data.index, pd.DatetimeIndex):
        mrsi = compute_multi_rsi(data).iloc[-1]
    else:
        mrsi = float('nan')

    price = data['close'].iloc[-1]
    if (
        short_ma > long_ma
        and (pd.isna(rsi) or rsi < 70)
        and sentiment_score >= 0
        and macro_score >= 0
        and onchain_score > 1.5
        and book_skew > 0.2
        and heatmap_ratio > 1.2
        and price < upper
        and direction >= 0
        and (pd.isna(anchor_low) or price > anchor_low)
        and (pd.isna(vwap) or price > vwap)
        and (pd.isna(obv) or obv > 0)
        and (pd.isna(wt) or wt > -50)
        and (pd.isna(mrsi) or mrsi < 30)
    ):
        return Signal('BUY')
    if (
        short_ma < long_ma
        and (pd.isna(rsi) or rsi > 30)
        and sentiment_score <= 0
        and macro_score <= 0
        and onchain_score < -1.5
        and book_skew < -0.2
        and heatmap_ratio < 0.8
        and price > lower
        and direction <= 0
        and (pd.isna(anchor_high) or price < anchor_high)
        and (pd.isna(vwap) or price < vwap)
        and (pd.isna(obv) or obv < 0)
        and (pd.isna(wt) or wt < 50)
        and (pd.isna(mrsi) or mrsi > 70)
    ):
        return Signal('SELL')
    return Signal('HOLD')
