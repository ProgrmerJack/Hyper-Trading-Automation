"""
Trading strategy classes.

This subpackage contains a collection of strategy implementations
inspired by the research report.  Each strategy encapsulates the
logic required to generate trading signals or orders based on
different sources of alpha, ranging from classical market making and
statistical arbitrage to event‑driven trading and machine learning.

While some classes here contain complete (though simplified)
implementations, others serve primarily as templates or stubs.  They
include detailed docstrings describing how a real implementation
would function.  This allows practitioners to extend the system
incrementally without having to start from scratch.

Available strategy classes
-------------------------

MarketMakerStrategy
    Implements a simplified Avellaneda–Stoikov market making model.
StatisticalArbitrageStrategy
    Executes pair trading or cross‑exchange arbitrage based on
    spread deviations.
TriangularArbitrageStrategy
    Exploits price discrepancies in three‑leg currency loops.
LatencyArbitrageStrategy
    Placeholder for ultra‑low latency arbitrage (unimplemented).
EventTradingStrategy
    Stub for news‑based or event driven trading strategies.
MLStrategy
    Base class for machine learning driven signals.
RLStrategy
    Placeholder for reinforcement learning policies controlling
    orders.
MetaStrategy
    Combines multiple underlying strategies and allocates capital
    dynamically.

See each module for further details.
"""

from .market_maker import MarketMakerStrategy, AvellanedaStoikov
from .stat_arb import StatisticalArbitrageStrategy, PairStatArb
from .triangular_arb import TriangularArbitrageStrategy, TriangularArb
from .latency_arb import LatencyArbitrageStrategy
from .event_trading import EventTradingStrategy
from .ml_strategy import MLStrategy, SimpleMLS
from .rl_strategy import RLStrategy
from .meta import MetaStrategy
from .breakout_donchian import DonchianBreakout
from .mean_reversion_ema import MeanReversionEMA
from .momentum_multi_tf import MomentumMultiTF

__all__ = [
    "MarketMakerStrategy",
    "StatisticalArbitrageStrategy",
    "TriangularArbitrageStrategy",
    "LatencyArbitrageStrategy",
    "EventTradingStrategy",
    "MLStrategy",
    "SimpleMLS",
    "RLStrategy",
    "MetaStrategy",
    "DonchianBreakout",
    "MeanReversionEMA",
    "MomentumMultiTF",
    "AvellanedaStoikov",
    "PairStatArb",
    "TriangularArb",
]
