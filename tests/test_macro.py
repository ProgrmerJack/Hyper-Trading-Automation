import pandas as pd
from hypertrader.utils.macro import compute_macro_score, compute_cardboard_zscore


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
