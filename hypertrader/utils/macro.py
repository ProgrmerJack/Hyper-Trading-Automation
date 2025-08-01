import pandas as pd


def _trend_score(series: pd.Series, window: int = 50, reverse: bool = False) -> float:
    ma = series.rolling(window=window).mean()
    if len(ma.dropna()) == 0:
        return 0.0
    score = 1 if series.iloc[-1] > ma.iloc[-1] else -1
    return -score if reverse else score


def compute_macro_score(dxy: pd.Series, rates: pd.Series, liquidity: pd.Series) -> float:
    """Combine macro indicators into a single score.

    A positive score favours crypto strength, negative score favours weakness.
    """
    scores = [
        _trend_score(dxy, reverse=True),            # strong dollar is bad for crypto
        _trend_score(rates, reverse=True),          # rising rates negative
        _trend_score(liquidity),                    # rising liquidity positive
    ]
    return sum(scores) / len(scores)
