import pandas as pd

from hypertrader.strategies.ml_strategy import (
    extract_features,
    train_model,
    ml_signal,
    cross_validate_model,
)


def test_train_and_predict():
    df = pd.DataFrame(
        {
            "open": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "high": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "low": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "close": [1, 2, 3, 2, 3, 4, 5, 6, 5, 6],
            "volume": [1] * 10,
        },
        index=pd.date_range("2024-01-01", periods=10, freq="h"),
    )
    model = train_model(df)
    sig = ml_signal(model, df)
    assert sig.action in {"BUY", "SELL", "HOLD"}
    assert 0.0 <= sig.probability <= 1.0


def test_cross_validate_model():
    df = pd.DataFrame(
        {
            "open": range(50),
            "high": range(50),
            "low": range(50),
            "close": list(range(25)) + list(range(25)),
            "volume": [1] * 50,
        },
        index=pd.date_range("2024-01-01", periods=50, freq="h"),
    )
    score = cross_validate_model(df, cv=3)
    assert 0 <= score <= 1
