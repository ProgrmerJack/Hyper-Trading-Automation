"""
Event‑driven backtesting engine.

This subpackage implements a minimal backtester that simulates
historical execution of trading strategies.  It feeds historical
market data to strategies via a connector, collects the orders
generated, and updates a simulated portfolio accordingly.  The
backtester is designed for small scale experiments and is not
intended to replicate all nuances of exchange behaviour (e.g., partial
fills, queue position).  However, it provides a framework for
evaluating strategies using realistic order book and trade data.

Key Components
--------------

Backtester
    Main class orchestrating the simulation.  It loops over
    timestamps, queries the connector for market data, calls the
    strategies and updates a portfolio.

Portfolio
    A simple class tracking cash and positions.  It supports
    executing buy and sell operations and computing portfolio value
    given current prices.

Users may extend this module by adding metrics, slippage models or
support for asynchronous event handling.
"""

from .engine import Backtester, Portfolio
from .advanced_engine import AdvancedBacktester

__all__ = ["Backtester", "Portfolio", "AdvancedBacktester"]
