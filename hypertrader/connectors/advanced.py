"""
Advanced simulation connector for realistic backtesting.

This module defines ``AdvancedSimulationConnector``, a subclass of
``SimulationConnector`` that more faithfully approximates exchange
microstructure for backtesting purposes.  It introduces several key
features absent from the base simulation:

* **Order latency:** limit orders are not active immediately but
  become eligible for matching only after a configurable number of
  trade ticks (simulating network or processing delays).
* **Queue position:** multiple orders at the same price are
  matched FIFO based on the order in which they become active.
* **Price crossing logic:** limit orders are filled only when
  synthetic trade prices cross their limit price.  Buy orders fill
  when the last trade price falls below or equals the limit; sell
  orders fill when the price rises above or equals the limit.
* **Partial fills:** by default this implementation fills
  orders completely when the price crosses.  Subclasses can
  override ``_match_order`` to implement partial fills based on
  trade quantity or book depth.

The advanced connector is intended for use with the backtesting
engine when a more realistic model of execution is required.  It
does not attempt to simulate every detail of an order matching
engine but provides enough fidelity to test whether strategies are
viable after accounting for latency and queue.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import datetime as _dt

from .exchange import SimulationConnector, Order


class AdvancedSimulationConnector(SimulationConnector):
    """An enhanced simulation connector modelling latency and queue position."""

    def __init__(self, historical_data: Dict[str, List[Tuple[_dt.datetime, float, float, str]]], latency_ticks: int = 0):
        """Initialise the advanced connector.

        Parameters
        ----------
        historical_data : dict
            Same as for :class:`SimulationConnector`.
        latency_ticks : int, optional
            Number of trade events to delay activation of limit orders.
            A value of zero means orders are active immediately.
        """
        super().__init__(historical_data)
        self.latency_ticks = max(0, latency_ticks)
        # For each symbol maintain a queue of pending limit orders.
        # Each entry is (ready_index, order_id).  The order becomes
        # eligible for matching once the processed trade index
        # reaches or exceeds ``ready_index``.
        self._activation_queue: Dict[str, List[Tuple[int, int]]] = {
            sym: [] for sym in historical_data
        }

    # ------------------------------------------------------------------
    # Order placement overrides

    def place_order(self, symbol: str, side: str, quantity: float, price: Optional[float] = None) -> Order:
        """Place a limit or market order with latency and queue modelling.

        Limit orders are not immediately available for matching.  They
        are scheduled to become active after ``latency_ticks`` trade
        events have been processed.  Market orders bypass latency and
        are executed instantly using the base class implementation.
        """
        # Use the base implementation to build the Order object and
        # possibly perform immediate fills for market orders.  The
        # base class also registers the order in the OMS and open
        # order tracking.
        order = super().place_order(symbol, side, quantity, price)
        # If limit order, schedule activation.
        if price is not None:
            # Determine the trade index at which the order becomes
            # eligible for matching.  We base this on the current
            # processed index for the symbol plus the latency.
            current_processed = self._processed_indices.get(symbol, 0)
            ready_index = current_processed + self.latency_ticks
            self._activation_queue[symbol].append((ready_index, order.order_id))
        return order

    # ------------------------------------------------------------------
    # Matching logic

    def process_open_orders(self, symbol: str, last_price: float) -> List[Tuple[Order, float, float]]:
        """Attempt to match and fill active limit orders for ``symbol``.

        This method should be called after processing each trade or
        synthetic price update.  It inspects the activation queue for
        orders whose ``ready_index`` is less than or equal to the
        current processed trade index.  Orders become active in FIFO
        order and are matched if the last trade price crosses the
        order's limit price.  When an order fills, it is removed
        from the open orders and the OMS updated accordingly.  The
        method returns a list of tuples ``(order, fill_price, qty)``
        representing orders that were filled during this call.

        Parameters
        ----------
        symbol : str
            Trading pair to process orders for.
        last_price : float
            The most recent trade price.

        Returns
        -------
        list of (Order, float, float)
            Each tuple contains the order object, the fill price and the
            quantity filled (equal to the order quantity for full fills).
        """
        fills: List[Tuple[Order, float, float]] = []
        # Determine the current processed index for the symbol.
        current_idx = self._processed_indices.get(symbol, 0)
        queue = self._activation_queue[symbol]
        # Orders that remain in queue after processing
        remaining: List[Tuple[int, int]] = []
        for ready_index, order_id in queue:
            # Only consider orders that have passed their latency
            if ready_index <= current_idx:
                order = self._open_orders.get(order_id)
                if order is None:
                    # Already filled or cancelled
                    continue
                # Determine if price crosses the limit
                if order.side == "buy":
                    # Buy limit executes if market price <= limit
                    if last_price <= (order.price or 0):
                        fill_qty = order.quantity
                        fill_price = last_price
                        self._fill_order(order, fill_price)
                        fills.append((order, fill_price, fill_qty))
                        continue  # do not add back to queue
                elif order.side == "sell":
                    # Sell limit executes if market price >= limit
                    if last_price >= (order.price or 0):
                        fill_qty = order.quantity
                        fill_price = last_price
                        self._fill_order(order, fill_price)
                        fills.append((order, fill_price, fill_qty))
                        continue
                # Price did not cross; keep in queue
                remaining.append((ready_index, order_id))
            else:
                # Not yet active; keep in queue
                remaining.append((ready_index, order_id))
        # Update queue to only include remaining orders
        self._activation_queue[symbol] = remaining
        return fills

    def _fill_order(self, order: Order, fill_price: float) -> None:
        """Helper to fill an order completely at the provided price.

        Updates the order state, notifies the OMS and removes it from
        the open orders map.
        """
        order.state = "FILLED"
        order.filled_qty = order.quantity
        order.avg_filled_price = fill_price
        # Update OMS if present
        if getattr(self, "oms", None) is not None:
            # Transition to FILLED in OMS and record the fill
            try:
                from ..execution.order_manager import OrderState
                self.oms.update_state(order.order_id, OrderState.FILLED)  # type: ignore
                self.oms.fill(order.order_id, order.quantity, fill_price)  # type: ignore
            except ImportError:
                pass  # OMS not available
        # Remove from open orders
        self._open_orders.pop(order.order_id, None)