"""
Meta strategy for combining multiple trading strategies.

The ``MetaStrategy`` manages a collection of sub‑strategies and
aggregates their order recommendations.  This allows the bot to run
multiple strategies concurrently on the same or different
instruments.  The meta strategy can optionally filter or throttle
orders using a risk management callback or dynamic capital
allocation logic.  For example, one might allocate more capital to
market making during calm periods and shift to arbitrage when
mispricings appear.

The current implementation simply loops through its constituent
strategies, collects their orders and returns them.  Order sizes
should already be scaled appropriately within each strategy.  To
prioritise or merge conflicting orders, override the
``aggregate_orders`` method.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Tuple


@dataclass
class MetaStrategy:
    """Combine multiple strategies into a unified interface.

    Parameters
    ----------
    strategies : list
        A list of strategy objects.  Each must implement an
        ``update`` method returning a list of orders.
    risk_callback : callable, optional
        A function ``risk_callback(order_list) -> List[Tuple[str, Any, float]]``
        that filters or modifies the aggregate orders prior to
        execution.  If ``None``, orders are returned as collected.
    """

    strategies: List[Any]
    risk_callback: Optional[Callable[[List[Tuple[str, Any, float]]], List[Tuple[str, Any, float]]]] = None

    def update(self, *args, **kwargs) -> List[Tuple[str, Any, float]]:
        """Call ``update`` on each strategy and combine their orders.

        Additional positional and keyword arguments are passed to
        every strategy.  This allows a single data feed to fan out
        across strategies that operate on the same symbol.  If a
        strategy requires different arguments, wrap it in a lambda or
        provide default values.

        Returns
        -------
        list
            A list of orders aggregated from all strategies.  The
            format is ``(side, symbol, quantity)``.
        """
        orders: List[Tuple[str, Any, float]] = []
        for strat in self.strategies:
            try:
                result = strat.update(*args, **kwargs)
                if result:
                    orders.extend(result)
            except Exception:
                # In production, log and continue
                continue
        # apply risk callback if provided
        if self.risk_callback is not None:
            orders = self.risk_callback(orders)
        return orders
