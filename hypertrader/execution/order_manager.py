from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict


class OrderState(Enum):
    """Lifecycle states for an order."""

    PENDING = auto()
    OPEN = auto()
    PARTIAL = auto()
    FILLED = auto()
    CANCELED = auto()


@dataclass(slots=True)
class Order:
    """Simple order representation used by :class:`OrderManager`."""

    id: str
    symbol: str
    side: str
    quantity: float
    filled: float = 0.0
    state: OrderState = field(default=OrderState.PENDING)


class OrderManager:
    """Track and update order state in response to execution events."""

    def __init__(self) -> None:
        self.orders: Dict[str, Order] = {}

    def submit(self, order: Order) -> None:
        """Register a new order in ``PENDING`` state."""
        self.orders[order.id] = order

    def on_ack(self, order_id: str) -> None:
        """Mark the order as acknowledged/open."""
        if order_id in self.orders:
            self.orders[order_id].state = OrderState.OPEN

    def on_fill(self, order_id: str, quantity: float) -> None:
        """Update filled quantity and state based on fills."""
        order = self.orders.get(order_id)
        if not order:
            return
        order.filled += quantity
        if order.filled >= order.quantity:
            order.state = OrderState.FILLED
        else:
            order.state = OrderState.PARTIAL

    def on_cancel(self, order_id: str) -> None:
        """Mark an order as canceled."""
        if order_id in self.orders:
            self.orders[order_id].state = OrderState.CANCELED

    def cancel_all(self) -> None:
        """Cancel all outstanding orders."""
        for order in self.orders.values():
            if order.state not in (OrderState.FILLED, OrderState.CANCELED):
                order.state = OrderState.CANCELED
