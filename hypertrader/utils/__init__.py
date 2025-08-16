"""
Utility functions and classes for hypertrader_plus.

This subpackage houses generic helpers that are shared across
connectors, indicators, strategies and the bot.  It re‑exports risk
management functions, anomaly detection and reinforcement learning
utilities from the original :mod:`hypertrader.utils` namespace and
includes additional wrappers where needed.  The aim is to centralise
common building blocks so that strategies remain concise and
readable.

Modules
-------

* :mod:`risk` – position sizing, leverage selection, trailing stops,
  drawdown throttles and RL throttles.
* :mod:`rl_utils` – reinforcement learning utilities for dynamic order
  sizing.  Includes a placeholder state scoring function.
* :mod:`anomaly` – entropy computations and regime classification.

For microstructure and technical indicators, see
``hypertrader_plus.indicators``.
"""

from .risk import (
    calculate_position_size,
    dynamic_leverage,
    trailing_stop,
    drawdown_throttle,
    drl_throttle,
)
from .rl_utils import dynamic_order_size, score_state
from .anomaly import compute_entropy, detect_entropy_regime

__all__ = [
    # risk management
    "calculate_position_size",
    "dynamic_leverage",
    "trailing_stop",
    "drawdown_throttle",
    "drl_throttle",
    # RL utilities
    "dynamic_order_size",
    "score_state",
    # anomaly
    "compute_entropy",
    "detect_entropy_regime",
]
