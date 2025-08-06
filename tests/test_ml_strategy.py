import numpy as np
import pandas as pd

from hypertrader.strategies.ml_strategy import (
    train_model,
    ml_signal,
    cross_validate_model,
    extract_features,
)


def _sample_df(n: int = 120) -> pd.DataFrame:
    close = pd.Series(np.sin(np.linspace(0, 20, n)) + 10)
    volume = pd.Series([100] * n)
    return pd.DataFrame({"close": close, "volume": volume})


def test_train_and_predict():
    df = _sample_df()
    model = train_model(df)
    sig = ml_signal(model, df)
    assert sig.action in {"BUY", "SELL", "HOLD"}
    assert 0.0 <= sig.probability <= 1.0


def test_cross_validate_model():
    df = _sample_df(240)
    score = cross_validate_model(df, cv=3)
    assert 0 <= score <= 1


def test_extract_features_contains_new_indicators():
    import numpy as np
    idx = pd.date_range("2020", periods=100, freq="1min")
    df = pd.DataFrame(
        {
            "close": np.linspace(1, 2, 100),
            "high": np.linspace(1.1, 2.1, 100),
            "low": np.linspace(0.9, 1.9, 100),
            "volume": [100] * 100,
        },
        index=idx,
    )
    feat = extract_features(df)
    for col in {"wavetrend", "multi_rsi", "poc_diff"}:
        assert col in feat.columns
