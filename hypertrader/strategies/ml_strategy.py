from __future__ import annotations

"""Simple machine learning-based strategy utilities."""

from dataclasses import dataclass
from typing import Optional

import pandas as pd
from sklearn.linear_model import LogisticRegression

from ..utils.features import compute_rsi, compute_ema


@dataclass
class MLSignal:
    action: str
    probability: float


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return dataframe of features for ML model."""
    features = pd.DataFrame(index=df.index)
    features["return"] = df["close"].pct_change().fillna(0)
    features["rsi"] = compute_rsi(df["close"]).fillna(0)
    features["ema_fast"] = compute_ema(df["close"], 10).fillna(method="bfill")
    features["ema_slow"] = compute_ema(df["close"], 30).fillna(method="bfill")
    features["ema_diff"] = features["ema_fast"] - features["ema_slow"]
    return features.dropna()


def train_model(df: pd.DataFrame) -> LogisticRegression:
    """Train a simple logistic regression model on historical data."""
    feat = extract_features(df)
    # Predict next period direction
    y = (df["close"].shift(-1).loc[feat.index] > df["close"].loc[feat.index]).astype(int)
    model = LogisticRegression(max_iter=200)
    model.fit(feat, y)
    return model


def ml_signal(model: LogisticRegression, df: pd.DataFrame) -> MLSignal:
    """Generate ML-based trading signal."""
    feat = extract_features(df).iloc[[-1]]
    prob = model.predict_proba(feat)[0, 1]
    if prob > 0.6:
        action = "BUY"
    elif prob < 0.4:
        action = "SELL"
    else:
        action = "HOLD"
    return MLSignal(action=action, probability=prob)
