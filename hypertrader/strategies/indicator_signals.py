from dataclasses import dataclass
from typing import Optional
import pandas as pd

from ..utils.features import (
    compute_ema,
    compute_rsi,
    compute_bollinger_bands,
    compute_supertrend,
    compute_anchored_vwap,
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
    else:
        anchor_high = anchor_low = float('nan')

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
    ):
        return Signal('SELL')
    return Signal('HOLD')
