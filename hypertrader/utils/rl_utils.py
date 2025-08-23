"""
Reinforcement‑learning utilities for adaptive trading.

This module exposes helper functions that implement simple
reinforcement‑learning inspired logic for adjusting order sizes in
response to market conditions.  It wraps the implementations from
``hypertrader.utils.rl_utils`` so they can be used within the
``hypertrader_plus`` package without introducing extra dependencies.

At present these functions use deterministic heuristics derived from
the state variables (model probability, order flow toxicity and
entropy regime) to scale the base position size.  They serve as a
placeholder for more sophisticated agents and are easily swapped out
once a proper RL model is trained on historical simulations.

Functions
---------

dynamic_order_size(prob_up, toxicity, regime, base_size, max_multiplier=2.0)
    Compute a scaled order size given current observations.  Increases
    size when conditions are favourable and decreases it otherwise.
score_state(prob_up, toxicity, regime)
    Compute a desirability score in [0, 1] summarising whether the
    current environment is conducive to trading.  Higher values
    indicate better opportunities.
"""

from __future__ import annotations

from typing import Literal


def score_state(
    prob_up: float,
    toxicity: float,
    regime: Literal["trending", "normal", "chaotic"],
) -> float:
    """Return a desirability score in the ``[0, 1]`` range.

    The function favours high probability of an upward move while
    penalising toxic order flow.  Simple adjustments are applied for
    market regime: ``trending`` slightly boosts the score, while
    ``chaotic`` reduces it.  The result is clipped to stay within
    bounds so callers can safely use it as a weighting factor.
    """

    score = prob_up - toxicity
    if regime == "trending":
        score += 0.1
    elif regime == "chaotic":
        score -= 0.1
    # Clamp to [0, 1]
    return max(0.0, min(1.0, score))


def dynamic_order_size(
    prob_up: float,
    toxicity: float,
    regime: Literal["trending", "normal", "chaotic"],
    base_size: float,
    max_multiplier: float = 2.0,
) -> float:
    """Scale ``base_size`` based on the current environment.

    The scaling factor is derived from :func:`score_state` and ranges
    between ``0`` and ``max_multiplier``.  A score of ``0.5`` yields the
    original ``base_size`` while higher/lower scores proportionally
    increase or decrease the order size.
    """

    desirability = score_state(prob_up, toxicity, regime)
    # Map score in [0,1] to [0, max_multiplier]
    multiplier = 1.0 + (desirability - 0.5) * 2.0 * (max_multiplier - 1.0)
    multiplier = max(0.0, min(max_multiplier, multiplier))
    return base_size * multiplier


__all__ = [
    "dynamic_order_size",
    "score_state",
]
