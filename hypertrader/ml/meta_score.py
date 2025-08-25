"""Utilities for combining multiple signals into a meta score and gating trades.

Trading strategies often rely on diverse sources of information.  To
avoid over‑reliance on any single indicator and to reduce false
positives, it is common practice to combine independent signals into
a unified meta score.  This module provides simple functions to
compute such a score from microstructure, technical, sentiment and
regime factors, as well as a gating function to decide whether the
overall context is favourable for taking a position.

The functions herein are intentionally lightweight.  Users can
substitute more sophisticated ensemble models (e.g. gradient boosted
trees or neural networks) if desired.  The default weightings are
equal across all inputs, but can be overridden via a dictionary.
"""

from __future__ import annotations

from typing import Dict, Mapping


def compute_meta_score(
    micro_score: float,
    tech_score: float,
    sentiment_score: float,
    regime_score: float,
    weights: Mapping[str, float] | None = None,
) -> float:
    """Compute a weighted meta score combining four factors.

    Parameters
    ----------
    micro_score:
        Normalised metric capturing short‑term order book or flow
        imbalance.  Positive values favour longs.
    tech_score:
        Aggregate output of technical indicator strategies.  Positive
        values favour longs.
    sentiment_score:
        Combined sentiment logit derived from news and social sources.
    regime_score:
        Forecasted change from a time‑series model.  Positive values
        imply an uptrend.
    weights:
        Optional mapping specifying the weight of each component.  Keys
        should include ``micro``, ``tech``, ``sentiment`` and
        ``regime``.  Missing keys default to equal weighting.  The
        weights are automatically normalised so they sum to one.

    Returns
    -------
    float
        Weighted sum of the inputs.  Higher values indicate a stronger
        bullish consensus, while negative values indicate bearish
        conditions.
    """
    default_weights = {"micro": 1.0, "tech": 1.0, "sentiment": 1.0, "regime": 1.0}
    if weights is not None:
        # copy to avoid mutating the caller's dict
        for k, v in weights.items():
            if k in default_weights:
                default_weights[k] = float(v)
    # Normalise weights
    total = sum(default_weights.values())
    if total == 0:
        total = 1.0
    w = {k: v / total for k, v in default_weights.items()}
    return (
        micro_score * w["micro"]
        + tech_score * w["tech"]
        + sentiment_score * w["sentiment"]
        + regime_score * w["regime"]
    )


def gate_entry(
    sentiment_score: float, regime_label: str, sentiment_threshold: float = 0.0
) -> bool:
    """Decide whether to enter a position based on sentiment and regime.

    The simplest gating approach requires that sentiment be above a
    threshold (indicating bullishness) and that the forecasted regime
    be an uptrend.  This reduces trading during adverse news regimes or
    when the market is expected to decline.

    Parameters
    ----------
    sentiment_score:
        Aggregated sentiment logit from FinBERT and social models.  A
        value above ``sentiment_threshold`` signals positive sentiment.
    regime_label:
        Output of :meth:`RegimeForecaster.classify_regime` ("uptrend" or
        "downtrend").
    sentiment_threshold:
        Threshold above which sentiment is considered positive.

    Returns
    -------
    bool
        ``True`` if the trade should be allowed, otherwise ``False``.
    """
    return bool(sentiment_score > sentiment_threshold and regime_label == "uptrend")
