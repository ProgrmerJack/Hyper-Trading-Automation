"""
Persistent order management system for hypertrader_plus.

This module defines a lightweight persistent order management system
used by connectors to track the lifecycle of orders across the
trading system.  The OMS maintains a mapping of order IDs to
``ManagedOrder`` objects, exposes methods to submit new orders,
update their state, record fills and cancel orders.  It also
supports cancel‑on‑disconnect semantics by exposing a ``cancel_all``
method which marks all known orders as ``CANCELLED``.

Orders progress through a simple set of states:

``NEW`` → ``ACK`` → ``PARTIAL`` → ``FILLED`` or ``CANCELLED``

Where ``NEW`` indicates the order has been created locally,
``ACK`` means the order has been acknowledged by the connector or
exchange, ``PARTIAL`` denotes a partially filled order and
``FILLED`` represents a fully filled order.  ``CANCELLED`` is a
terminal state indicating that the order will receive no further
fills.  The OMS does not enforce state transitions strictly but
clients are expected to follow them to maintain consistency.

For simplicity, this OMS stores data in memory only.  In
production one would persist orders to a database or durable
storage to survive process restarts.  The interface is intentionally
minimal but can be extended with callbacks, logging or other
functionality as needed.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Dict, Optional


class OrderState(enum.Enum):
    """Enumerated states for an order's lifecycle."""

    NEW = "NEW"
    ACK = "ACK"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"


@dataclass
class ManagedOrder:
    """Order representation tracked by the persistent OMS.

    Parameters
    ----------
    order_id : int
        Unique identifier of the order.
    order : object
        Reference to the order as returned from the connector.  The
        OMS does not enforce any particular type here but expects
        that ``order.order_id`` matches ``order_id``.
    state : OrderState
        Current state of the order.
    filled_qty : float
        Total quantity filled so far.
    avg_price : Optional[float]
        Volume‑weighted average price of executed fills.  ``None``
        indicates no fills have occurred yet.
    """

    order_id: int
    order: object
    state: OrderState
    filled_qty: float = 0.0
    avg_price: Optional[float] = None

    def record_fill(self, qty: float, price: float) -> None:
        """Update filled quantity and average price.

        This method updates the managed order's ``filled_qty`` and
        ``avg_price`` given a new fill.  It will also transition
        ``state`` to ``PARTIAL`` or ``FILLED`` depending on whether
        the full order size has been executed.  The caller must
        ensure that ``qty`` does not exceed the remaining quantity.

        Parameters
        ----------
        qty : float
            Quantity filled in this fill.
        price : float
            Execution price for this fill.
        """
        if qty <= 0:
            return
        # Update volume‑weighted average price
        if self.filled_qty == 0:
            self.avg_price = price
        else:
            assert self.avg_price is not None  # for type checkers
            total_value = self.avg_price * self.filled_qty + price * qty
            self.avg_price = total_value / (self.filled_qty + qty)
        self.filled_qty += qty
        # Determine new state based on filled quantity versus order quantity
        # Only update to PARTIAL or FILLED if current state is not CANCELLED
        if self.state != OrderState.CANCELLED:
            # Access the order's original quantity if available
            original_qty = getattr(self.order, "quantity", None)
            if original_qty is not None:
                if self.filled_qty >= original_qty:
                    self.state = OrderState.FILLED
                else:
                    self.state = OrderState.PARTIAL


class PersistentOMS:
    """A simple persistent order management system.

    The OMS tracks managed orders in an in‑memory dictionary and
    exposes methods to submit new orders, update their state,
    record fills and cancel orders.  It does not handle
    concurrency; if used in a multi‑threaded environment, callers
    should synchronise access to the OMS instance.
    """

    def __init__(self) -> None:
        self._orders: Dict[int, ManagedOrder] = {}

    # ------------------------------------------------------------------
    # Order lifecycle operations

    def submit(self, order: object) -> None:
        """Register a new order with the OMS.

        The order is stored in the internal registry in the ``NEW`` state.
        If an order with the same ID already exists, it is silently
        overwritten.
        """
        mo = ManagedOrder(order_id=getattr(order, "order_id"), order=order, state=OrderState.NEW)
        self._orders[mo.order_id] = mo

    def update_state(self, order_id: int, new_state: OrderState) -> None:
        """Change the state of an existing order.

        If the order is not present in the OMS, this call is a no‑op.
        The OMS does not enforce state transition rules; clients must
        decide how and when to transition states.
        """
        mo = self._orders.get(order_id)
        if mo is not None:
            mo.state = new_state

    def fill(self, order_id: int, qty: float, price: float) -> None:
        """Record a fill for the given order ID.

        Parameters
        ----------
        order_id : int
            Identifier of the order to update.
        qty : float
            Quantity filled.
        price : float
            Fill price.
        """
        mo = self._orders.get(order_id)
        if mo is not None:
            mo.record_fill(qty, price)

    def cancel(self, order_id: int) -> None:
        """Mark an order as cancelled.

        This does not remove the order from the registry but sets
        its state to ``CANCELLED`` so that further fills are ignored.
        """
        mo = self._orders.get(order_id)
        if mo is not None:
            mo.state = OrderState.CANCELLED

    def cancel_all(self) -> None:
        """Cancel all tracked orders.

        Iterates through all managed orders and sets their state to
        ``CANCELLED``.  It does not remove orders from the registry.
        """
        for mo in self._orders.values():
            mo.state = OrderState.CANCELLED

    # ------------------------------------------------------------------
    # Query operations

    def get(self, order_id: int) -> Optional[ManagedOrder]:
        """Retrieve the managed order for the given ID, if present."""
        return self._orders.get(order_id)

    def all_orders(self) -> Dict[int, ManagedOrder]:
        """Return a snapshot of all managed orders."""
        return dict(self._orders)

    def clear(self) -> None:
        """Remove all orders from the OMS.

        This does not affect orders on the exchange or connector;
        it simply resets the in‑memory registry.  Use with caution.
        """
        self._orders.clear()