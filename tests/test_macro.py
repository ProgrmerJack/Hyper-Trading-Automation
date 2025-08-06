import pandas as pd
from hypertrader.utils.macro import (
    compute_macro_score,
    compute_cardboard_zscore,
    compute_risk_tolerance,
)


def test_compute_macro_score():
    dxy = pd.Series([100]*60)
    rates = pd.Series([5]*60)
    liquidity = pd.Series([10]*60)
    score = compute_macro_score(dxy, rates, liquidity)
    assert isinstance(score, float)


def test_compute_cardboard_zscore():
    series = pd.Series([1]*100)
    z = compute_cardboard_zscore(series)
    assert abs(z) < 1e-9


def test_compute_macro_score_with_cardboard():
    dxy = pd.Series([100]*120)
    rates = pd.Series([5]*120)
    liquidity = pd.Series([10]*120)
    cardboard = pd.Series(range(120))
    score = compute_macro_score(dxy, rates, liquidity, cardboard)
    assert score > 0


def test_compute_risk_tolerance():
    idx = pd.date_range("2024-01-01", periods=60, freq="D")
    liquidity = pd.Series(range(60), index=idx)
    yield_spread = pd.Series(range(60), index=idx)
    vix = pd.Series([30 - i * 0.1 for i in range(60)], index=idx)
    silver = pd.Series(range(1, 61), index=idx)
    gold = pd.Series([100] * 60, index=idx)
    score = compute_risk_tolerance(liquidity, yield_spread, vix, silver, gold)
    assert isinstance(score, pd.Series)
    assert len(score) == 60
    assert score.iloc[-1] > 0
