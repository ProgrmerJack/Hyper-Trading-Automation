"""Backtesting utilities for time‑series models and trading strategies.

Over‑fitting is a pervasive problem in algorithmic trading.  Models
that perform well on historical data often fail to generalise due to
look‑ahead bias and the non‑stationary nature of financial markets.
This module implements two routines aimed at mitigating these issues:

* **Purged K‑Fold cross‑validation** with an optional embargo.  Unlike
  standard cross‑validation, purging removes training samples that
  overlap with the test set to prevent information leakage.  The
  embargo further skips a specified number of observations on either
  side of the test fold.
* **Walk‑forward backtesting**, where a model is repeatedly trained
  on an expanding window and evaluated on the subsequent out‑of‑sample
  period.  This mirrors the incremental nature of real trading and
  produces realistic performance metrics such as the Sharpe ratio and
  maximum drawdown.

These functions are generic and can be applied to any estimator
supporting ``fit`` and ``predict`` or ``predict_proba`` methods.
"""

from __future__ import annotations

from typing import Iterable, Iterator, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def purged_kfold_cv(n_splits: int, embargo: int = 0) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Generate train/test indices for purged K‑Fold cross‑validation.

    Parameters
    ----------
    n_splits:
        Number of folds.  Must be at least 2.
    embargo:
        Number of samples to exclude before and after each test fold to
        reduce leakage.  Expressed in number of observations, not time.

    Yields
    ------
    tuple[np.ndarray, np.ndarray]
        Tuple containing training indices and test indices for each fold.

    Notes
    -----
    The implementation does not shuffle the data; it assumes the
    observations are in chronological order.  The returned indices can
    be used directly with pandas DataFrames via ``iloc``.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    kf = KFold(n_splits=n_splits, shuffle=False)
    n_samples = None
    # The outer generator defers to this closure; we capture the data
    # length when the returned generator is consumed.
    def splitter(X: Sequence) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        nonlocal n_samples
        indices = np.arange(len(X))
        n_samples = len(indices)
        for train_idx, test_idx in kf.split(indices):
            # Determine embargoed region
            start = max(0, test_idx[0] - embargo)
            end = min(n_samples, test_idx[-1] + embargo + 1)
            # Create mask to exclude the embargoed region
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[start:end] = False
            yield indices[train_mask], test_idx
    return splitter  # type: ignore[return-value]


def walk_forward_backtest(
    model,
    features: pd.DataFrame,
    labels: pd.Series,
    initial_window: int,
    step: int,
) -> Tuple[float, float]:
    """Perform a walk‑forward backtest of a model on time‑series data.

    A model is retrained on an expanding window of historical data and
    evaluated on a fixed horizon ``step`` ahead.  Performance metrics
    such as the average Sharpe ratio and average minimum drawdown are
    computed to assess robustness.

    Parameters
    ----------
    model:
        Estimator supporting ``fit`` and ``predict`` or ``predict_proba``.
    features:
        DataFrame of explanatory variables indexed chronologically.
    labels:
        Series of binary labels or returns aligned with ``features``.
    initial_window:
        Number of initial observations used to train the first model.
    step:
        Number of observations to predict in each walk‑forward iteration.

    Returns
    -------
    tuple[float, float]
        Mean Sharpe ratio and mean drawdown across all iterations.  A
        higher Sharpe and lower drawdown indicate better performance.
    """
    if initial_window <= 0 or step <= 0:
        raise ValueError("initial_window and step must be positive")
    if len(features) != len(labels):
        raise ValueError("features and labels must have the same length")
    start = initial_window
    sharpe_ratios: list[float] = []
    drawdowns: list[float] = []
    while start < len(features):
        train_X = features.iloc[:start]
        train_y = labels.iloc[:start]
        test_X = features.iloc[start : start + step]
        test_y = labels.iloc[start : start + step]
        if len(test_y) == 0:
            break
        # Fit the model on the training window
        model.fit(train_X, train_y)
        # Obtain predictions; support both predict_proba and predict
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(test_X)[:, 1]
        else:
            pred = model.predict(test_X)
            prob = np.asarray(pred, dtype=float)
        # Compute simple returns: long if probability > 0.5 else short
        signals = (prob > 0.5).astype(int) * 2 - 1
        # Map binary labels {0,1} to returns {-1,1}
        actual = (test_y.astype(int) * 2) - 1
        # Daily P&L = signal * actual return
        pnl = signals * actual
        # Cumulative P&L for Sharpe and drawdown calculations
        cum_pnl = pnl.cumsum()
        mean = float(np.mean(cum_pnl))
        std = float(np.std(cum_pnl))
        sharpe = mean / (std + 1e-9)
        sharpe_ratios.append(sharpe)
        drawdown = float(np.min(cum_pnl))
        drawdowns.append(drawdown)
        start += step
    # Compute average metrics
    avg_sharpe = float(np.mean(sharpe_ratios)) if sharpe_ratios else 0.0
    avg_drawdown = float(np.mean(drawdowns)) if drawdowns else 0.0
    return avg_sharpe, avg_drawdown
