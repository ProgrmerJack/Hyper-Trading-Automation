"""Machine learning-based strategy utilities and implementations.

This module provides both utility functions for training sklearn models
and a class-based strategy implementation that uses machine learning
to predict short-term price movements. It includes feature extraction,
model training, cross-validation, and real-time strategy execution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Tuple

import numpy as np
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
    compute_ichimoku,
    compute_parabolic_sar,
    compute_keltner_channels,
    compute_cci,
    compute_fibonacci_retracements,
    compute_atr,
    compute_twap,
    compute_cumulative_delta,
)

from ..utils.macro import compute_risk_tolerance


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))



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
        features["atr"] = compute_atr(df).fillna(0)
        ichimoku = compute_ichimoku(df)
        features["tenkan"] = ichimoku["tenkan"].fillna(0)
        features["kijun"] = ichimoku["kijun"].fillna(0)
        features["psar"] = compute_parabolic_sar(df).fillna(0)
        keltner = compute_keltner_channels(df)
        features["kelt_width"] = (keltner["upper"] - keltner["lower"]).fillna(0)
        features["cci"] = compute_cci(df).fillna(0)
        fib = compute_fibonacci_retracements(df)
        features["fib_dist"] = (df["close"] - fib["level_0.618"]).fillna(0)
    if isinstance(df.index, pd.DatetimeIndex):
        features["multi_rsi"] = compute_multi_rsi(df).fillna(0)
    if "volume" in df:
        vwap = compute_vwap(df)
        features["vwap_ratio"] = (df["close"] / vwap - 1).fillna(0)
        features["obv"] = compute_obv(df).fillna(0)
        poc = compute_vpvr_poc(df)
        features["poc_diff"] = (df["close"] - poc).fillna(0)
        features["twap"] = compute_twap(df).fillna(0)
        if {"buy_vol", "sell_vol"}.issubset(df.columns):
            features["cum_delta"] = compute_cumulative_delta(df).fillna(0)
    if {"inflows", "outflows"}.issubset(df.columns):
        features["net_flow"] = compute_exchange_netflow(df).fillna(0)

    required_cols = {"liquidity", "yield_spread", "vix", "silver", "gold"}
    if required_cols.issubset(df.columns):
        risk = compute_risk_tolerance(
            df["liquidity"],
            df["yield_spread"],
            df["vix"],
            df["silver"],
            df["gold"],
        )
        features["risk_tolerance"] = risk

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
    """Generate ML-based trading signal with SHAP explainability."""
    from ..utils.risk import shap_explain
    
    feat = extract_features(df).iloc[[-1]]
    prob = model.predict_proba(feat)[0, 1]
    
    # Add SHAP explainability for model interpretation
    try:
        shap_values = shap_explain(model, feat)
    except Exception:
        shap_values = None
    
    if prob > 0.6:
        action = "BUY"
    elif prob < 0.4:
        action = "SELL"
    else:
        action = "HOLD"
    return MLSignal(action=action, probability=prob)


def simple_ml_signal(df: pd.DataFrame, lookback: int = 50, weight: float = 20.0) -> Tuple[int, float]:
    """Generate simple momentum-based ML signal."""
    if len(df) < lookback + 1:
        return 0, 0.5
    sub = df.iloc[-lookback:].copy()
    r = sub['close'].pct_change().fillna(0.0).values
    x = np.nan_to_num(r[-10:]).sum()
    p = 1 / (1 + np.exp(-weight * x))
    sig = 1 if p > 0.55 else -1 if p < 0.45 else 0
    return sig, p


@dataclass
class MLStrategy:
    """Machine learning-based strategy using logistic regression.

    Parameters
    ----------
    symbol : str
        Instrument to trade.
    window : int
        Number of past observations used to compute features.
    buy_threshold : float
        Probability above which to trigger a buy.
    sell_threshold : float
        Probability below which to trigger a sell.
    base_order_size : float
        Nominal order size adjusted by dynamic sizing.
    weights : tuple of float, optional
        Coefficients for logistic regression (momentum, toxicity, bias).
    """

    symbol: str
    window: int = 50
    buy_threshold: float = 0.55
    sell_threshold: float = 0.45
    base_order_size: float = 1.0
    weights: Tuple[float, float, float] = (10.0, -5.0, 0.0)
    # stateful series
    prices: List[float] = field(default_factory=list, init=False)
    trades: List[dict] = field(default_factory=list, init=False)

    def predict_prob_up(self, current_price: float, toxicity: float) -> float:
        """Compute probability of upward move using logistic regression.

        Uses price momentum and order flow toxicity as features.
        """
        from ..indicators.technical import sma
        
        # Compute momentum
        if len(self.prices) < 2:
            momentum = 0.0
        else:
            avg = sma(self.prices, self.window)
            momentum = current_price - avg
        w1, w2, b = self.weights
        z = w1 * momentum + w2 * toxicity + b
        return _sigmoid(z)

    def update(self, current_price: float, recent_trades: Iterable[dict]) -> List[Tuple[str, float, float]]:
        """Update strategy with new price and trades, return orders.

        Parameters
        ----------
        current_price : float
            Latest trade or mid price.
        recent_trades : iterable of dict
            Recent trade data for toxicity computation.

        Returns
        -------
        list of tuple
            Orders in format (side, price, quantity).
        """
        from ..indicators.microstructure import flow_toxicity, detect_entropy_regime
        from ..utils.rl_utils import dynamic_order_size
        
        self.prices.append(current_price)
        # compute toxicity
        tox = flow_toxicity(list(recent_trades), window=min(len(recent_trades), 100))
        # compute probability
        prob_up = self.predict_prob_up(current_price, tox)
        # Determine regime from recent price changes
        deltas = []
        if len(self.prices) > 1:
            for i in range(max(0, len(self.prices) - 20), len(self.prices) - 1):
                direction = 1 if self.prices[i + 1] > self.prices[i] else 0
                deltas.append(direction)
        regime = detect_entropy_regime(deltas) if deltas else "normal"
        # Determine size based on RL dynamic sizing
        size = dynamic_order_size(prob_up, tox, regime, self.base_order_size, max_multiplier=2.0)
        orders: List[Tuple[str, float, float]] = []
        if prob_up > self.buy_threshold:
            orders.append(("buy", current_price, size))
        elif prob_up < self.sell_threshold:
            orders.append(("sell", current_price, size))
        return orders


@dataclass
class SimpleMLS:
    """Minimal ML strategy using momentum-based logistic regression."""
    lookback: int = 50
    weight: float = 20.0
    
    def update(self, df: pd.DataFrame):
        """Update with DataFrame, return (signal, confidence, metadata)."""
        if len(df) < self.lookback + 1:
            return 0, 0.5, {}
        sub = df.iloc[-self.lookback:].copy()
        r = sub['close'].pct_change().fillna(0.0).values
        x = np.nan_to_num(r[-10:]).sum()
        p = 1 / (1 + np.exp(-self.weight * x))
        sig = 1 if p > 0.55 else -1 if p < 0.45 else 0
        return sig, p, {'x': float(x), 'p': float(p)}
