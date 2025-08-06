import pandas as pd

from hypertrader.utils.features import (
    compute_rsi,
    compute_moving_average,
    compute_ema,
    compute_atr,
    compute_bollinger_bands,
    compute_supertrend,
    compute_anchored_vwap,
    onchain_zscore,
    order_skew,
    dom_heatmap_ratio,
)


def test_compute_moving_average():
    series = pd.Series([1, 2, 3, 4, 5])
    ma = compute_moving_average(series, 2)
    assert ma.iloc[-1] == 4.5


def test_compute_rsi():
    prices = pd.Series([1, 2, 3, 2, 1, 2, 3])
    rsi = compute_rsi(prices, period=2)
    assert not rsi.isna().all()


def test_compute_atr():
    df = pd.DataFrame(
        {
            "high": [2, 3, 4],
            "low": [1, 1, 2],
            "close": [1.5, 2.5, 3.5],
        }
    )
    atr = compute_atr(df, period=2)
    assert len(atr) == 3


def test_compute_bollinger_bands():
    series = pd.Series([1] * 20)
    bands = compute_bollinger_bands(series, window=5)
    assert list(bands.columns) == ["ma", "upper", "lower"]


def test_compute_ema():
    series = pd.Series([1, 2, 3, 4, 5])
    ema = compute_ema(series, span=3)
    assert round(ema.iloc[-1], 2) > 0


def test_compute_supertrend():
    df = pd.DataFrame(
        {
            "high": [2, 3, 4, 5],
            "low": [1, 2, 3, 4],
            "close": [1.5, 2.5, 3.5, 4.5],
        }
    )
    st = compute_supertrend(df, period=2)
    assert list(st.columns) == ["supertrend", "direction"]
    assert len(st) == 4


def test_compute_anchored_vwap():
    df = pd.DataFrame(
        {
            "high": [1, 3, 2],
            "low": [0, 1, 1],
            "close": [1, 2, 3],
            "volume": [10, 20, 30],
        }
    )
    vwap = compute_anchored_vwap(df, anchor="high")
    assert vwap.dropna().iloc[-1] > 0


def test_onchain_zscore():
    df = pd.DataFrame({"gas": [10] * 40})
    z = onchain_zscore(df, window=10)
    assert (z == 0).all()


def test_order_skew():
    book = {"bids": [[1, 2], [0.9, 1]], "asks": [[1.1, 1], [1.2, 2]]}
    skew = order_skew(book, depth=2)
    assert round(skew, 2) == 0.0


def test_dom_heatmap_ratio():
    book = {"bids": [[1, 5]], "asks": [[1.1, 1]]}
    ratio = dom_heatmap_ratio(book, layers=1)
    assert ratio == 5.0
