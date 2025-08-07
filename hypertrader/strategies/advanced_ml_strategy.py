from __future__ import annotations

"""Advanced machine learning utilities for hyper-trading.

This module introduces a more complex ML pipeline including feature
engineering utilities, a transformer based classifier and a lightweight
reinforcement learning environment.  The goal is to provide an easily
extensible baseline for high frequency trading research without imposing
heavy runtime costs on users that do not require the functionality.
"""

from typing import Iterable, Tuple

import numpy as np
import pandas as pd

# scikit-learn components are required for the feature engineering stage.
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

# Optional heavy dependencies are imported lazily inside functions to avoid
# import errors during unit tests where these libraries may not be available.


def advanced_feature_engineering(
    df: pd.DataFrame,
    lag_windows: Iterable[int] = (1, 5, 10),
    n_pca: int = 20,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Generate an enriched feature set for complex models.

    Parameters
    ----------
    df:
        DataFrame containing at least ``close`` prices and a ``target`` column
        representing the label used for supervised learning.
    lag_windows:
        Sequence of lags used to compute past returns and volatility.
    n_pca:
        Number of principal components returned after dimensionality
        reduction.

    Returns
    -------
    (DataFrame, ndarray)
        Tuple of processed feature matrix and corresponding target array.
    """

    df = df.copy()
    if "target" not in df:
        raise ValueError("DataFrame must contain a 'target' column")

    # ------------------------------------------------------------------
    # 1) Feature expansion
    # ------------------------------------------------------------------
    for lag in lag_windows:
        df[f"return_lag_{lag}"] = df["close"].pct_change(lag)
        df[f"vol_lag_{lag}"] = df["close"].rolling(lag).std()

    # Simple GARCH(1,1) like volatility estimate
    df["garch_vol"] = np.sqrt(
        0.1 * (df["return_lag_1"] ** 2) + 0.9 * (df["vol_lag_1"] ** 2)
    )

    # Sentiment score using FinBERT if news headlines are available.
    if "news_headlines" in df:
        try:
            from transformers import pipeline

            sentiment_pipe = pipeline("sentiment-analysis", model="ProsusAI/finbert")
            df["sentiment_score"] = [
                sentiment_pipe(text)[0]["score"] if isinstance(text, str) and text else 0.0
                for text in df["news_headlines"]
            ]
        except Exception:
            # If the transformer model is not available we simply default to zero.
            df["sentiment_score"] = 0.0
    else:
        df["sentiment_score"] = 0.0

    # Normalised on-chain gas fee information if supplied.
    if "gas_fees" in df:
        gas = df["gas_fees"].astype(float)
        df["gas_fee_z"] = (gas - gas.mean()) / gas.std(ddof=0)
    else:
        df["gas_fee_z"] = 0.0

    # ------------------------------------------------------------------
    # 2) Feature selection and dimensionality reduction
    # ------------------------------------------------------------------
    X = df.drop(columns=["target"]).fillna(0.0)
    y = df["target"].astype(int).values

    if X.shape[1] > 0:
        mi_scores = mutual_info_classif(X.values, y)
        top_features = np.argsort(mi_scores)[-min(50, X.shape[1]) :]
        X_selected = X.iloc[:, top_features]
    else:
        X_selected = X

    # PCA for compression
    n_components = min(n_pca, X_selected.shape[1]) or 1
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X_selected)

    # ------------------------------------------------------------------
    # 3) Normalisation and optional data augmentation
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X_reduced)

    try:  # SMOTE is optional and only used when available
        from imblearn.over_sampling import SMOTE

        smote = SMOTE(random_state=42)
        X_aug, y_aug = smote.fit_resample(X_norm, y)
    except Exception:
        X_aug, y_aug = X_norm, y

    columns = [f"feat_{i}" for i in range(X_aug.shape[1])]
    return pd.DataFrame(X_aug, columns=columns), y_aug


# --------------------------------------------------------------------------
# Transformer based classifier
# --------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - handled gracefully in environments without torch
    torch = None  # type: ignore
    nn = None  # type: ignore


class TradingTransformer(nn.Module):
    """Minimal transformer model for directional price prediction."""

    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 8, num_layers: int = 4, dropout: float = 0.1) -> None:
        if torch is None:
            raise ImportError("PyTorch is required for TradingTransformer")
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # x shape: [batch, seq_len, features]
        x = self.embedding(x).permute(1, 0, 2)  # [seq_len, batch, d_model]
        out = self.transformer(x).mean(dim=0)
        return torch.sigmoid(self.fc(out))


class FocalLoss(nn.Module):
    """Binary focal loss for imbalanced classification."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25) -> None:
        if torch is None:
            raise ImportError("PyTorch is required for FocalLoss")
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.bce = nn.BCELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bce = self.bce(pred, target)
        pt = torch.exp(-bce)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        return loss


# --------------------------------------------------------------------------
# Reinforcement learning environment using Stable-Baselines3
# --------------------------------------------------------------------------
try:
    from gym import Env, spaces  # type: ignore
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
except Exception:  # pragma: no cover - optional dependency
    Env = object  # type: ignore
    spaces = None  # type: ignore
    PPO = None  # type: ignore
    make_vec_env = None  # type: ignore


class TradingEnv(Env):  # type: ignore[misc]
    """Simple trading environment producing discrete actions."""

    def __init__(self, data: pd.DataFrame) -> None:
        if spaces is None:
            raise ImportError("gym is required for TradingEnv")
        super().__init__()
        self.data = data.reset_index(drop=True)
        self.current_step = 0
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(data.shape[1],))
        self.action_space = spaces.Discrete(3)  # buy/sell/hold

    def reset(self):  # type: ignore[override]
        self.current_step = 0
        return self.data.iloc[0].values

    def step(self, action: int):  # type: ignore[override]
        reward = self._calculate_reward(action)
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        obs = self.data.iloc[self.current_step].values
        return obs, reward, done, {}

    def _calculate_reward(self, action: int) -> float:
        """Placeholder reward function based on price difference."""
        if self.current_step >= len(self.data) - 1:
            return 0.0
        price_diff = self.data.iloc[self.current_step + 1][0] - self.data.iloc[self.current_step][0]
        if action == 1:  # buy
            return float(price_diff)
        if action == 2:  # sell
            return float(-price_diff)
        return 0.0


def train_transformer(X: np.ndarray, y: np.ndarray, epochs: int = 100, lr: float = 1e-3) -> TradingTransformer:
    """Train ``TradingTransformer`` using the provided dataset."""
    if torch is None:
        raise ImportError("PyTorch is required for train_transformer")
    model = TradingTransformer(input_dim=X.shape[-1])
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = FocalLoss()
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    for _ in range(epochs):
        preds = model(X_t)
        loss = criterion(preds, y_t)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    return model


def train_drl(data: pd.DataFrame, total_timesteps: int = 100_000):
    """Train a PPO agent on ``TradingEnv``."""
    if PPO is None or make_vec_env is None:
        raise ImportError("stable-baselines3 and gym are required for train_drl")
    env = make_vec_env(TradingEnv, n_envs=4, env_kwargs={"data": data})
    model = PPO("MlpPolicy", env, learning_rate=1e-4, n_steps=2048)
    model.learn(total_timesteps=total_timesteps)
    return model


def walk_forward_train(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> TradingTransformer:
    """Perform walk-forward training returning the last fitted model."""
    from sklearn.model_selection import TimeSeriesSplit

    splitter = TimeSeriesSplit(n_splits=n_splits)
    model: TradingTransformer | None = None
    for train_idx, _ in splitter.split(X):
        X_train, y_train = X[train_idx], y[train_idx]
        model = train_transformer(X_train, y_train)
    if model is None:
        raise ValueError("Training failed - empty dataset")
    return model


def hyper_tune(objective, n_trials: int = 50):
    """Run Optuna optimisation returning the best parameters."""
    try:
        import optuna
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("Optuna is required for hyper_tune") from exc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def ml_ensemble(models: list, X: np.ndarray) -> np.ndarray:
    """Soft voting ensemble returning averaged probabilities."""
    probs = [model.predict_proba(X)[:, 1] for model in models]
    return np.mean(probs, axis=0)


def hybrid_signal(ensemble_prob: float, rsi: float, threshold: float = 0.6) -> str:
    """Combine ML probability with a simple RSI filter."""
    return "buy" if ensemble_prob > threshold and rsi < 30 else "hold"


__all__ = [
    "advanced_feature_engineering",
    "TradingTransformer",
    "FocalLoss",
    "TradingEnv",
    "train_transformer",
    "train_drl",
    "walk_forward_train",
    "hyper_tune",
    "ml_ensemble",
    "hybrid_signal",
]
