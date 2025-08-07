from __future__ import annotations

"""Utilities for simulating prop-trading style funding challenges."""

from dataclasses import dataclass
from typing import Callable, Dict


@dataclass
class PropResult:
    """Result of a prop challenge simulation."""

    scaled_capital: float
    passed: bool


def prop_challenge(
    backtest_fn: Callable[[], Dict[str, float]],
    capital: float = 100.0,
    target_profit: float = 0.1,
    max_dd: float = 0.05,
    scale_factor: int = 100,
) -> PropResult:
    """Evaluate a strategy against simple prop-firm style rules.

    Parameters
    ----------
    backtest_fn: callable returning dict
        Function that returns ``{"profit": float, "mdd": float}`` summarising
        backtest performance.
    capital: float
        Starting capital used for the simulation.
    target_profit: float
        Profit target expressed as fraction of capital (e.g. ``0.1`` for 10%).
    max_dd: float
        Maximum allowed drawdown expressed as fraction of capital.
    scale_factor: int
        Multiplier applied to capital on success.
    """

    stats = backtest_fn()
    passed = stats.get("profit", 0) > target_profit and stats.get("mdd", 1) < max_dd
    scaled = capital * scale_factor if passed else capital
    return PropResult(scaled_capital=scaled, passed=passed)
