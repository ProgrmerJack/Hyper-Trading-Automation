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
    compute_ichimoku,
    compute_parabolic_sar,
    compute_keltner_channels,
    compute_cci,
    compute_fibonacci_retracements,
    compute_atr,
)


@dataclass
class Signal:
    action: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    volume: float = 0.0
    confidence: float = 0.6


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
        atr = compute_atr(data).iloc[-1]
        ichimoku = compute_ichimoku(data)
        tenkan = ichimoku['tenkan'].iloc[-1]
        kijun = ichimoku['kijun'].iloc[-1]
        psar = compute_parabolic_sar(data).iloc[-1]
        keltner = compute_keltner_channels(data)
        kelt_upper = keltner['upper'].iloc[-1]
        kelt_lower = keltner['lower'].iloc[-1]
        cci = compute_cci(data).iloc[-1]
        fib = compute_fibonacci_retracements(data)
        fib_618 = fib['level_0.618'].iloc[-1]
    else:
        anchor_high = anchor_low = vwap = obv = atr = float('nan')
        tenkan = kijun = psar = kelt_upper = kelt_lower = cci = fib_618 = float('nan')

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
        and (pd.isna(tenkan) or price > tenkan)
        and (pd.isna(psar) or price > psar)
        and (pd.isna(kelt_lower) or price > kelt_lower)
        and (pd.isna(cci) or cci > -100)
        and (pd.isna(fib_618) or price > fib_618 * 0.99)
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
        and (pd.isna(tenkan) or price < tenkan)
        and (pd.isna(psar) or price < psar)
        and (pd.isna(kelt_upper) or price < kelt_upper)
        and (pd.isna(cci) or cci < 100)
        and (pd.isna(fib_618) or price < fib_618 * 1.01)
    ):
        return Signal('SELL')
    return Signal('HOLD')
