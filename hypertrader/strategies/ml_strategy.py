from __future__ import annotations

"""Simple machine learning-based strategy utilities."""

from dataclasses import dataclass
from typing import Optional

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from ..utils.features import (
    compute_rsi,
    compute_ema,
    compute_macd,
    compute_vwap,
    compute_obv,
    compute_adx,
    compute_stochastic,
    compute_roc,
    compute_exchange_netflow,
    compute_volatility_cluster,
    compute_ai_momentum,
    compute_wavetrend,
    compute_multi_rsi,
    compute_vpvr_poc,
)


@dataclass
class MLSignal:
    action: str
    probability: float


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return dataframe of features for ML model."""
    features = pd.DataFrame(index=df.index)
    features["return"] = df["close"].pct_change().fillna(0)
    features["rsi"] = compute_rsi(df["close"]).fillna(0)
    # Use backfill to handle initial NaN values without relying on deprecated API
    features["ema_fast"] = compute_ema(df["close"], 10).bfill()
    features["ema_slow"] = compute_ema(df["close"], 30).bfill()
    features["ema_diff"] = features["ema_fast"] - features["ema_slow"]
    macd = compute_macd(df["close"])
    features["macd_hist"] = macd["histogram"].fillna(0)
    features["roc"] = compute_roc(df["close"]).fillna(0)
    features["ai_momo"] = compute_ai_momentum(df["close"]).fillna(0)
    features["vol_cluster"] = compute_volatility_cluster(df).fillna(0)
    if {"high", "low"}.issubset(df.columns):
        features["adx"] = compute_adx(df).fillna(0)
        stoch = compute_stochastic(df)
        features["stoch_k"] = stoch["k"].fillna(0)
        features["stoch_d"] = stoch["d"].fillna(0)
        features["wavetrend"] = compute_wavetrend(df).fillna(0)
    if isinstance(df.index, pd.DatetimeIndex):
        features["multi_rsi"] = compute_multi_rsi(df).fillna(0)
    if "volume" in df:
        vwap = compute_vwap(df)
        features["vwap_ratio"] = (df["close"] / vwap - 1).fillna(0)
        features["obv"] = compute_obv(df).fillna(0)
        poc = compute_vpvr_poc(df)
        features["poc_diff"] = (df["close"] - poc).fillna(0)
    if {"inflows", "outflows"}.issubset(df.columns):
        features["net_flow"] = compute_exchange_netflow(df).fillna(0)
    return features.dropna()


def train_model(df: pd.DataFrame) -> LogisticRegression:
    """Train a simple logistic regression model on historical data."""
    feat = extract_features(df)
    # Predict next period direction
    y = (df["close"].shift(-1).loc[feat.index] > df["close"].loc[feat.index]).astype(
        int
    )
    model = LogisticRegression(max_iter=200, class_weight="balanced")
    model.fit(feat, y)
    return model


def cross_validate_model(df: pd.DataFrame, cv: int = 5) -> float:
    """Return mean cross-validation accuracy for logistic regression."""
    feat = extract_features(df)
    y = (df["close"].shift(-1).loc[feat.index] > df["close"].loc[feat.index]).astype(
        int
    )
    if y.nunique() < 2:
        return 0.0
    model = LogisticRegression(max_iter=200, class_weight="balanced")
    try:
        scores = cross_val_score(model, feat, y, cv=min(cv, y.value_counts().min()))
    except ValueError:
        return 0.0
    return float(scores.mean())


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
