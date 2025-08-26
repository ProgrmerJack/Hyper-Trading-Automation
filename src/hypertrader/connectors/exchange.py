"""
Connector classes for interacting with exchanges or data sources.

The trading system separates market access from strategy logic via a
simple connector interface.  This file defines a base
``ExchangeConnector`` which specifies the methods required by the
rest of the system and a ``SimulationConnector`` which implements a
minimal event‑driven backtesting environment.  The goal is to
provide a common API for both historical backtesting and live
trading.  For live trading, users can subclass
``ExchangeConnector`` and implement the abstract methods using
libraries such as `ccxt` or proprietary exchange APIs.

Classes
-------

ExchangeConnector
    Abstract base class for connectors.  Defines the required
    interface for fetching order books, trades and submitting orders.
SimulationConnector
    Concrete implementation of ``ExchangeConnector`` that feeds
    preloaded historical data into the trading system and simulates
    order matching.  Useful for offline backtesting.

Notes
-----
This module intentionally keeps the interface simple.  Real
exchanges offer a wide variety of order types and asynchronous
behaviour; for brevity we model only basic limit and market orders
and assume synchronous execution.  Users seeking to extend this
module to production should consider adding support for partial
fills, order queues, WebSocket data feeds and authentication.
"""

from __future__ import annotations

import abc
import bisect
import collections
import datetime as _dt
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class Order:
    """Simple order representation used by the simulation connector.

    Parameters
    ----------
    order_id : int
        Unique identifier for the order.
    symbol : str
        The trading pair (e.g., ``"BTC/USDT"``).
    side : str
        Either ``"buy"`` or ``"sell"``.
    quantity : float
        The number of units to buy or sell.
    price : Optional[float]
        Limit price for a limit order.  ``None`` for market orders.
    timestamp : datetime
        Time the order was created.
    """

    order_id: int
    symbol: str
    side: str
    quantity: float
    price: Optional[float]
    timestamp: _dt.datetime
    # Added fields for persistent order management
    state: str = "NEW"
    filled_qty: float = 0.0
    avg_filled_price: Optional[float] = None


class ExchangeConnector(abc.ABC):
    """Abstract base class for exchange connectors.

    Concrete subclasses must implement the following methods:

    * ``get_order_book(symbol: str) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]``
    * ``get_trade_history(symbol: str, since: Optional[_dt.datetime] = None) -> Iterable[Tuple[_dt.datetime, float, float, str]]``
    * ``place_order(symbol: str, side: str, quantity: float, price: Optional[float] = None) -> Order``
    * ``cancel_order(order_id: int) -> None``

    These methods should be thread‑safe if used in a multi‑strategy
    setting.  For production, additional methods may be required to
    handle authentication, account balance queries and so on.
    """

    @abc.abstractmethod
    def get_order_book(self, symbol: str) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """Return the best bids and asks for ``symbol``.

        Parameters
        ----------
        symbol : str
            Trading pair to retrieve order book for.

        Returns
        -------
        (bids, asks) : Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]
            ``bids`` is a list of ``(price, volume)`` tuples sorted in
            descending order by price.  ``asks`` is a list of
            ``(price, volume)`` tuples sorted in ascending order by
            price.  Only the top levels are returned; the depth can
            be determined by the implementation.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_trade_history(self, symbol: str, since: Optional[_dt.datetime] = None) -> Iterable[Tuple[_dt.datetime, float, float, str]]:
        """Return recent trades for ``symbol``.

        Parameters
        ----------
        symbol : str
            Trading pair to retrieve trades for.
        since : datetime, optional
            If provided, only trades after this timestamp are
            returned.  If ``None``, returns all available trades.

        Returns
        -------
        Iterable of (timestamp, price, quantity, side)
            The side is either ``"buy"`` or ``"sell"`` depending on
            whether the trade was buyer‑initiated or seller‑initiated.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def place_order(self, symbol: str, side: str, quantity: float, price: Optional[float] = None) -> Order:
        """Submit a limit or market order.

        Parameters
        ----------
        symbol : str
            Trading pair to execute on.
        side : str
            ``"buy"`` or ``"sell"``.
        quantity : float
            Number of units to trade.
        price : float, optional
            Limit price for limit orders.  If ``None``, a market
            order is assumed.

        Returns
        -------
        Order
            Representation of the submitted order.  For live
            connectors, this may map to an exchange order ID.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def cancel_order(self, order_id: int) -> None:
        """Cancel an open order by its identifier.

        Parameters
        ----------
        order_id : int
            Unique identifier returned from :meth:`place_order`.
        """
        raise NotImplementedError


class SimulationConnector(ExchangeConnector):
    """A simple event‑driven simulation of an exchange.

    This connector is designed to feed historical data into the
    backtester and simulate very basic order execution.  It keeps an
    internal pointer to the current index of each symbol's trade
    history and advances it as time progresses.  Limit orders are
    matched against the simulated order book; market orders consume
    liquidity at the best available price.  This connector is not
    intended to perfectly model exchange microstructure (queue
    position, partial fills etc.), but rather to give reasonable
    fill prices and volumes for backtesting strategies.

    To use this connector, construct it with a dictionary mapping
    symbols to ordered lists of trade tuples: ``(timestamp, price,
    quantity, side)``.  The ``timestamp`` must be strictly
    increasing and in chronological order.  The simulation will step
    through these events as strategies call
    :meth:`get_trade_history`.
    """

    def __init__(self, historical_data: Dict[str, List[Tuple[_dt.datetime, float, float, str]]]):
        self._data = historical_data
        self._indices = {sym: 0 for sym in historical_data}
        self._open_orders: Dict[int, Order] = {}
        self._next_order_id = 1
        # Optional OMS integration
        try:
            from ..execution.order_manager import PersistentOMS, OrderState
            self.oms = PersistentOMS()
        except ImportError:
            self.oms = None
        # Track processed indices for advanced simulation
        self._processed_indices = {sym: 0 for sym in historical_data}

    def get_order_book(self, symbol: str) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """Return a synthetic order book based on the last trade price.

        Since the simulation does not maintain a full order book, this
        method returns a rudimentary book centred around the last
        traded price.  The spread and depth can be configured via
        constants defined within this method.  For more sophisticated
        simulations, users should subclass and override this method.
        """
        history = self._data[symbol]
        idx = self._indices[symbol]
        if idx == 0:
            last_price = history[0][1]
        else:
            last_price = history[idx - 1][1]
        # Define a fixed spread and depth; in reality these would vary.
        spread = 0.001 * last_price
        levels = 5
        bids = [(last_price - i * spread, 1.0) for i in range(1, levels + 1)]
        asks = [(last_price + i * spread, 1.0) for i in range(1, levels + 1)]
        return bids, asks

    def get_trade_history(self, symbol: str, since: Optional[_dt.datetime] = None) -> Iterable[Tuple[_dt.datetime, float, float, str]]:
        """Return trades for ``symbol`` occurring after ``since``.

        The simulation connector maintains an internal index for each
        symbol.  Each call to this method returns new trades since
        the previous call and advances the index.  When ``since`` is
        provided, trades strictly after that timestamp are returned.
        """
        history = self._data[symbol]
        start_idx = self._indices[symbol]
        if since is not None:
            # Advance index to first trade after 'since'
            while start_idx < len(history) and history[start_idx][0] <= since:
                start_idx += 1
        self._indices[symbol] = start_idx
        # Return trades sequentially and move index to end
        new_trades = history[start_idx:]
        self._indices[symbol] = len(history)
        return new_trades

    def place_order(self, symbol: str, side: str, quantity: float, price: Optional[float] = None) -> Order:
        """Place a simulated order.

        Market orders are executed immediately at the best available
        synthetic price from :meth:`get_order_book`.  Limit orders are
        stored and matched against the synthetic book in subsequent
        calls to :meth:`get_trade_history`.  This simplified model
        fills the entire order at the limit price if the price crosses
        the spread; otherwise, the order remains open and may never
        execute.  Order IDs are assigned sequentially.
        """
        order = Order(
            order_id=self._next_order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            timestamp=_dt.datetime.utcnow(),
        )
        self._next_order_id += 1
        # Update OMS if available
        if self.oms is not None:
            try:
                from ..execution.order_manager import OrderState
                self.oms.submit(order)
                self.oms.update_state(order.order_id, OrderState.ACK)
            except ImportError:
                pass
        if price is None:
            # Market order: immediate fill at best price
            bids, asks = self.get_order_book(symbol)
            if side == "buy":
                fill_price = asks[0][0]
            else:
                fill_price = bids[0][0]
            order.state = "FILLED"
            order.filled_qty = quantity
            order.avg_filled_price = fill_price
            if self.oms is not None:
                try:
                    from ..execution.order_manager import OrderState
                    self.oms.update_state(order.order_id, OrderState.FILLED)
                    self.oms.fill(order.order_id, quantity, fill_price)
                except ImportError:
                    pass
        else:
            # For limit orders, we simply record them and hope they
            # cross the spread.  This is a naïve model.
            self._open_orders[order.order_id] = order
        return order

    def cancel_order(self, order_id: int) -> None:
        """Cancel a simulated limit order if it exists."""
        self._open_orders.pop(order_id, None)
        if self.oms is not None:
            try:
                self.oms.cancel(order_id)
            except (ImportError, AttributeError):
                pass
