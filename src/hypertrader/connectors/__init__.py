"""
hypertrader_plus.connectors
===========================

This subpackage contains classes used to interact with external data
sources and exchanges.  Each connector abstracts away the details of
data retrieval and order submission so that the rest of the trading
system can remain agnostic to the underlying API.  Two example
connectors are provided:

* :class:`~hypertrader_plus.connectors.exchange.ExchangeConnector` is a
  base class that defines the common interface.  It exposes methods
  for fetching live or historical market data (order books, trades)
  and for placing or cancelling orders.  Concrete subclasses
  implement these methods for specific environments.
* :class:`~hypertrader_plus.connectors.exchange.SimulationConnector` is a
  lightweight backtesting connector.  It feeds historical data to
  strategies and simulates basic order execution.  This allows
  strategies to be validated in an event‑driven backtester without
  requiring access to a live exchange.

Connectors can be extended to support live trading through popular
libraries such as `ccxt` or exchange‑specific WebSocket APIs.  See
``exchange.py`` for implementation details and subclassing
instructions.
"""

from .exchange import ExchangeConnector, SimulationConnector, Order
from .advanced import AdvancedSimulationConnector

__all__ = ["ExchangeConnector", "SimulationConnector", "Order", "AdvancedSimulationConnector"]
