"""
Advanced backtesting engine for hypertrader.

The ``AdvancedBacktester`` builds upon the simple event‑driven
backtester by integrating the ``AdvancedSimulationConnector`` which
models latency, queue position and partial fills.  It also
incorporates fee and slippage gating so that orders are submitted
only when the expected edge exceeds transaction costs.  The engine
updates a portfolio in response to fills reported by the connector
and records PnL over time.

Usage example::

    from hypertrader.connectors.advanced import AdvancedSimulationConnector
    from hypertrader.strategies.market_maker import MarketMakerStrategy
    from hypertrader.backtester.advanced_engine import AdvancedBacktester

    connector = AdvancedSimulationConnector(historical_data, latency_ticks=2)
    strategy = MarketMakerStrategy(symbol="BTC/USDT", gamma=0.1, kappa=1.0, sigma=0.02, base_order_size=0.01)
    bt = AdvancedBacktester(connector, [strategy], start_cash=1000.0)
    results = bt.run("BTC/USDT")
    print(results["pnl"])

This class is independent of the live trading bot but shares the
strategy interface.  It should provide more realistic performance
estimates than the simple backtester.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple, Any

from ..connectors.advanced import AdvancedSimulationConnector
from ..utils.risk import fee_slippage_gate
from ..connectors.exchange import Order


@dataclass
class Portfolio:
    """Portfolio model with fee accounting.

    Attributes
    ----------
    cash : float
        Available cash balance.
    positions : dict
        Map from symbol to quantity held.
    fees_paid : float
        Cumulative trading fees paid.
    """

    cash: float
    positions: Dict[str, float] = field(default_factory=dict)
    fees_paid: float = 0.0

    def buy(self, symbol: str, price: float, qty: float, fee_rate: float) -> None:
        cost = price * qty
        fee = cost * fee_rate
        self.cash -= (cost + fee)
        self.fees_paid += fee
        self.positions[symbol] = self.positions.get(symbol, 0.0) + qty

    def sell(self, symbol: str, price: float, qty: float, fee_rate: float) -> None:
        proceeds = price * qty
        fee = proceeds * fee_rate
        self.cash += (proceeds - fee)
        self.fees_paid += fee
        self.positions[symbol] = self.positions.get(symbol, 0.0) - qty

    def value(self, prices: Dict[str, float]) -> float:
        val = self.cash
        for sym, qty in self.positions.items():
            val += qty * prices.get(sym, 0.0)
        return val


class AdvancedBacktester:
    """Event‑driven backtester using an advanced simulation connector."""

    def __init__(self, connector: AdvancedSimulationConnector, strategies: List[Any], start_cash: float = 10_000.0, fee_rate: float = 0.0005, slippage_rate: float = 0.0002):
        self.connector = connector
        self.strategies = strategies
        self.portfolio = Portfolio(cash=start_cash)
        self.prices: Dict[str, float] = {}
        self.pnl_history: List[Tuple[_dt.datetime, float]] = []
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        # Maintain a mapping of pending order IDs to side and quantity
        # so that fills can update the portfolio appropriately
        self.pending_orders: Dict[int, Tuple[str, float]] = {}

    def run(self, symbol: str) -> Dict[str, Any]:
        """Execute the backtest on a single symbol.

        Processes each trade event sequentially, updates strategies,
        submits orders, matches them via the connector and updates
        the portfolio accordingly.  It records the portfolio value
        after each trade for PnL analysis.

        Returns
        -------
        dict
            Contains final portfolio, PnL time series and trades executed.
        """
        history = self.connector._data[symbol]
        trades_executed: List[Tuple[_dt.datetime, str, float, float]] = []
        # For gating: we will use mid price = last trade price.  For limit orders
        # we call the gating function before submission.
        for idx, (ts, price, qty, side_trade) in enumerate(history):
            # Save the current price
            self.prices[symbol] = price
            # Update processed index so that latency activation knows where we are
            self.connector._processed_indices[symbol] = idx
            # Feed strategies
            for strat in self.strategies:
                orders = []
                try:
                    if hasattr(strat, "generate_orders"):
                        order_book = {"bids": [(price, 1.0)], "asks": [(price, 1.0)]}
                        orders = strat.generate_orders(order_book)
                    elif hasattr(strat, "update"):
                        # Determine argument signature.  Some strategies expect price only.
                        if strat.__class__.__name__ == "MLStrategy":
                            orders = strat.update(price, [])
                        else:
                            orders = strat.update(price)
                except Exception:
                    orders = []
                # For each order, apply gating and submit
                for side_order, order_price, order_qty in orders:
                    # Determine the execution price to test gating
                    tgt_price = order_price if order_price is not None else price
                    current_price = price
                    # Only submit if expected edge exceeds costs
                    if not fee_slippage_gate(current_price, tgt_price, fee_rate=self.fee_rate, slippage_rate=self.slippage_rate):
                        continue
                    # Place the order via the connector
                    order_obj: Order = self.connector.place_order(symbol, side_order, order_qty, order_price)
                    # Market orders fill immediately; update portfolio
                    if order_obj.price is None:
                        fill_price = order_obj.avg_filled_price or price
                        qty_filled = order_obj.filled_qty or order_qty
                        if side_order == "buy":
                            self.portfolio.buy(symbol, fill_price, qty_filled, self.fee_rate)
                        else:
                            self.portfolio.sell(symbol, fill_price, qty_filled, self.fee_rate)
                        trades_executed.append((ts, side_order, qty_filled, fill_price))
                    else:
                        # Limit order: record in pending orders for later fills
                        self.pending_orders[order_obj.order_id] = (side_order, order_qty)
            # After submitting new orders, process existing open orders for fills
            fills = self.connector.process_open_orders(symbol, price)
            for order_obj, fill_price, fill_qty in fills:
                # Determine side from pending map
                info = self.pending_orders.pop(order_obj.order_id, None)
                if info is None:
                    # Unknown order; skip
                    continue
                side_order, order_qty = info
                if side_order == "buy":
                    self.portfolio.buy(symbol, fill_price, fill_qty, self.fee_rate)
                else:
                    self.portfolio.sell(symbol, fill_price, fill_qty, self.fee_rate)
                trades_executed.append((ts, side_order, fill_qty, fill_price))
            # Mark current portfolio value
            current_value = self.portfolio.value(self.prices)
            self.pnl_history.append((ts, current_value))
        return {
            "portfolio": self.portfolio,
            "pnl": self.pnl_history,
            "trades": trades_executed,
        }