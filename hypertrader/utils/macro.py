import pandas as pd
import numpy as np


def _trend_score(series: pd.Series, window: int = 50, reverse: bool = False) -> float:
    ma = series.rolling(window=window).mean()
    if len(ma.dropna()) == 0:
        return 0.0
    score = 1 if series.iloc[-1] > ma.iloc[-1] else -1
    return -score if reverse else score


def compute_cardboard_zscore(cardboard: pd.Series, window: int = 90) -> float:
    """Compute z-score of cardboard production as a proxy for manufacturing trends."""
    mean = cardboard.rolling(window).mean()
    std = cardboard.rolling(window).std()
    z = (cardboard - mean) / std
    value = z.iloc[-1]
    if pd.isna(value) or not np.isfinite(value):
        return 0.0
    return float(value)


def compute_macro_score(
    dxy: pd.Series,
    rates: pd.Series,
    liquidity: pd.Series,
    cardboard: pd.Series | None = None,
) -> float:
    """Combine macro indicators into a single score.

    A positive score favours crypto strength, negative score favours weakness.
    """
    scores = [
        _trend_score(dxy, reverse=True),            # strong dollar is bad for crypto
        _trend_score(rates, reverse=True),          # rising rates negative
        _trend_score(liquidity),                    # rising liquidity positive
    ]
    if cardboard is not None:
        cb_z = compute_cardboard_zscore(cardboard)
        scores.append(1 if cb_z > 0 else -1)
    return sum(scores) / len(scores)
