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

from hypertrader.utils.rl_utils import dynamic_order_size as _dynamic_order_size, score_state as _score_state


def dynamic_order_size(
    prob_up: float,
    toxicity: float,
    regime: Literal["trending", "normal", "chaotic"],
    base_size: float,
    max_multiplier: float = 2.0,
) -> float:
    """Wrapper for :func:`hypertrader.utils.rl_utils.dynamic_order_size`.

    See that function for full documentation.
    """
    return _dynamic_order_size(prob_up, toxicity, regime, base_size, max_multiplier)


def score_state(prob_up: float, toxicity: float, regime: Literal["trending", "normal", "chaotic"]) -> float:
    """Wrapper for :func:`hypertrader.utils.rl_utils.score_state`.  See original docs.
    """
    return _score_state(prob_up, toxicity, regime)


__all__ = [
    "dynamic_order_size",
    "score_state",
]
