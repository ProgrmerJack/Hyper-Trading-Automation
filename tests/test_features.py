import pandas as pd

from hypertrader.utils.features import (
    compute_rsi,
    compute_moving_average,
    compute_ema,
    compute_atr,
    compute_bollinger_bands,
    compute_supertrend,
    compute_anchored_vwap,
    compute_vwap,
    compute_obv,
    compute_adx,
    compute_stochastic,
    compute_roc,
    compute_twap,
    compute_cumulative_delta,
    compute_cci,
    compute_keltner_channels,
    compute_wavetrend,
    compute_multi_rsi,
    compute_vpvr_poc,
    compute_ichimoku,
    compute_parabolic_sar,
    onchain_zscore,
    order_skew,
    dom_heatmap_ratio,
    compute_exchange_netflow,
    compute_volatility_cluster,
    compute_fibonacci_retracements,
    compute_ai_momentum,
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


def test_compute_vwap_and_obv():
    df = pd.DataFrame(
        {
            "close": [1, 2, 1, 2],
            "volume": [10, 20, 30, 40],
        }
    )
    vwap = compute_vwap(df)
    obv = compute_obv(df)
    assert vwap.iloc[-1] > 0
    assert obv.iloc[-1] != 0


def test_compute_adx_and_stochastic():
    df = pd.DataFrame(
        {
            "high": [1, 2, 3, 4, 5],
            "low": [0, 1, 2, 3, 4],
            "close": [0.5, 1.5, 2.5, 3.5, 4.5],
        }
    )
    adx = compute_adx(df, period=2)
    stoch = compute_stochastic(df, k_period=3, d_period=2)
    assert len(adx) == len(df)
    assert stoch["k"].iloc[-1] <= 100


def test_compute_roc_and_twap():
    series = pd.Series([1, 2, 3, 4, 5])
    roc = compute_roc(series, period=1)
    assert round(roc.iloc[-1], 2) == 25.0
    df = pd.DataFrame({"close": [1, 2, 3]}, index=pd.date_range("2020", periods=3, freq="1T"))
    twap = compute_twap(df)
    assert twap.iloc[-1] == df["close"].expanding().mean().iloc[-1]


def test_compute_cumulative_delta_and_cci_keltner():
    df = pd.DataFrame(
        {
            "buy_vol": [1, 2, 3],
            "sell_vol": [0, 1, 1],
            "high": [1, 2, 3],
            "low": [0, 1, 2],
            "close": [0.5, 1.5, 2.5],
        }
    )
    cum_delta = compute_cumulative_delta(df)
    cci = compute_cci(df, period=2)
    keltner = compute_keltner_channels(df, ema_period=2, atr_period=2)
    assert cum_delta.iloc[-1] > 0
    assert len(cci) == len(df)
    assert list(keltner.columns) == ["ema", "upper", "lower"]


def test_exchange_netflow_and_vol_cluster():
    df = pd.DataFrame(
        {
            "close": [1, 1.1, 1.2, 1.15, 1.18],
            "inflows": [10, 20, 15, 10, 5],
            "outflows": [5, 15, 20, 30, 25],
        }
    )
    net = compute_exchange_netflow(df)
    assert net.iloc[-1] == 20
    cluster = compute_volatility_cluster(df, window=2)
    assert len(cluster) == len(df)


def test_fibonacci_and_ai_momentum():
    df = pd.DataFrame(
        {
            "high": [1, 2, 3, 4, 5],
            "low": [0, 1, 1.5, 2, 2.5],
        }
    )
    fib = compute_fibonacci_retracements(df, window=5)
    assert "level_0.618" in fib.columns
    series = pd.Series(range(10))
    ai = compute_ai_momentum(series, period=5)
    assert ai.iloc[-1] > 0


def test_wavetrend_multi_rsi_vpvr_and_ichimoku_parabolic():
    df = pd.DataFrame(
        {
            "high": [2, 3, 4, 5, 6],
            "low": [1, 2, 3, 4, 5],
            "close": [1.5, 2.5, 3.5, 4.5, 5.5],
            "volume": [10, 20, 30, 40, 50],
        },
        index=pd.date_range("2020", periods=5, freq="1min"),
    )
    wt = compute_wavetrend(df)
    assert len(wt) == len(df)
    mrsi = compute_multi_rsi(df[["close"]])
    assert mrsi.between(0, 100).all()
    poc = compute_vpvr_poc(df, bins=2)
    assert df["low"].min() <= poc <= df["high"].max()
    ichi = compute_ichimoku(df)
    assert "tenkan" in ichi.columns
    sar = compute_parabolic_sar(df)
    assert len(sar) == len(df)
