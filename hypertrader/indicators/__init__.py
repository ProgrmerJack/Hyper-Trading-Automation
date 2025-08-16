"""
hypertrader_plus.indicators
===========================

This subpackage groups together functions for computing technical
indicators (moving averages, oscillators, volatility measures, etc.)
and microstructure indicators (order book imbalance, microprice,
toxicity, entropy, iceberg detection) that are crucial for high
frequency trading.  Many of these functions wrap implementations
from the sister ``hypertrader`` package for convenience and
reusability, while others are implemented here directly.

The indicators are split into two modules:

* :mod:`technical` provides conventional price/volume indicators such
  as EMA, RSI, MACD, Bollinger Bands and ATR.  These are useful for
  gauging broader market trends and momentum.
* :mod:`microstructure` focuses on limit order book features and
  order flow metrics.  Functions here include the microprice,
  simplified VPIN (volume–synchronised probability of informed
  trading), Shannon entropy of price movements and a heuristic
  iceberg detector.

These tools can be imported individually or via the module names.  For
example::

    from hypertrader_plus.indicators.technical import ema, rsi
    from hypertrader_plus.indicators.microstructure import compute_microprice

Please consult the documentation within each module for usage
examples and mathematical definitions.
"""

from .technical import (
    ema,
    sma,
    rsi,
    macd,
    atr,
    bollinger_bands,
    supertrend,
    anchored_vwap,
    obv,
    wavetrend,
    multi_rsi,
)
from .microstructure import (
    compute_microprice,
    flow_toxicity,
    compute_entropy,
    detect_iceberg,
    detect_entropy_regime,
)

__all__ = [
    # technical
    "ema",
    "sma",
    "rsi",
    "macd",
    "atr",
    "bollinger_bands",
    "supertrend",
    "anchored_vwap",
    "obv",
    "wavetrend",
    "multi_rsi",
    # microstructure
    "compute_microprice",
    "flow_toxicity",
    "compute_entropy",
    "detect_iceberg",
    "detect_entropy_regime",
]
