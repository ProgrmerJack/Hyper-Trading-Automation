import pandas as pd
from hypertrader.utils.macro import compute_macro_score


def test_compute_macro_score():
    dxy = pd.Series([100]*60)
    rates = pd.Series([5]*60)
    liquidity = pd.Series([10]*60)
    score = compute_macro_score(dxy, rates, liquidity)
    assert isinstance(score, float)
