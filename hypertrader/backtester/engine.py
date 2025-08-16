"""
Event‑driven backtesting engine for hypertrader_plus.

This module defines a small backtester that can be used to evaluate
strategies offline using historical trade data.  The backtester
operates by consuming sequential trade ticks from a
``SimulationConnector``, invoking strategies with the latest price
information, executing orders against a simplified price model and
updating a portfolio.  The engine is intentionally straightforward
and does not model partial fills, order queues or fees; for more
realistic simulation those aspects should be added by subclassing.

Example usage::

    from hypertrader_plus.connectors import SimulationConnector
    from hypertrader_plus.strategies import MarketMakerStrategy
    from hypertrader_plus.backtester import Backtester

    # load historical data as {symbol: [(timestamp, price, qty, side), ...]}
    connector = SimulationConnector(historical_data)
    strategy = MarketMakerStrategy("BTC/USDT", gamma=0.1, kappa=1.0, sigma=0.02, base_order_size=0.01)
    bt = Backtester(connector, [strategy], start_cash=1000.0)
    results = bt.run("BTC/USDT")
    print(results["pnl"])

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple, Any
import datetime as _dt

from ..connectors.exchange import SimulationConnector


@dataclass
class Portfolio:
    """Simple portfolio for backtesting.

    Tracks cash and positions in various symbols.  The portfolio
    assumes all trades are executed at the provided mid price and does
    not consider slippage or fees.  Cash is denominated in the quote
    currency of the traded instruments (e.g., USDT).

    Attributes
    ----------
    cash : float
        Available cash balance.
    positions : dict
        Map from symbol to quantity held.  Positive values denote
        long positions; negative values denote shorts.
    """

    cash: float
    positions: Dict[str, float] = field(default_factory=dict)

    def buy(self, symbol: str, price: float, qty: float) -> None:
        """Increase position in ``symbol`` by ``qty`` and decrease cash."""
        cost = price * qty
        self.cash -= cost
        self.positions[symbol] = self.positions.get(symbol, 0.0) + qty

    def sell(self, symbol: str, price: float, qty: float) -> None:
        """Decrease position in ``symbol`` by ``qty`` and increase cash."""
        proceeds = price * qty
        self.cash += proceeds
        self.positions[symbol] = self.positions.get(symbol, 0.0) - qty

    def value(self, prices: Dict[str, float]) -> float:
        """Compute total portfolio value given current prices."""
        val = self.cash
        for sym, qty in self.positions.items():
            price = prices.get(sym, 0.0)
            val += qty * price
        return val


class Backtester:
    """Run an event‑driven backtest over historical trade data.

    Parameters
    ----------
    connector : SimulationConnector
        Provides historical trades and synthetic order books.  Must be
        prepopulated with data for all symbols of interest.
    strategies : list
        List of strategy objects to run.  Each must implement an
        ``update`` method that accepts a price (or other arguments)
        and returns a list of orders ``(side, price, quantity)``.
    start_cash : float, optional
        Initial cash balance.  Default is 10_000.0.
    """

    def __init__(self, connector: SimulationConnector, strategies: List[Any], start_cash: float = 10_000.0):
        self.connector = connector
        self.strategies = strategies
        self.portfolio = Portfolio(cash=start_cash)
        # Track last price per symbol for mark to market
        self.prices: Dict[str, float] = {}
        # Store results
        self.pnl_history: List[Tuple[_dt.datetime, float]] = []

    def run(self, symbol: str) -> Dict[str, Any]:
        """Execute the backtest on a single symbol.

        This method iterates over all trades in the historical data
        provided to the connector.  For each trade it updates the
        strategies with the latest price, executes orders and records
        portfolio value.

        Parameters
        ----------
        symbol : str
            The symbol to trade.  Must be present in the connector's
            historical data.

        Returns
        -------
        dict
            Contains the final portfolio, a PnL time series and the
            list of trades executed.
        """
        history = self.connector._data[symbol]
        trades_executed: List[Tuple[_dt.datetime, str, float, float]] = []
        for ts, price, qty, side in history:
            # Save current price for mark to market
            self.prices[symbol] = price
            # Feed strategies
            for strat in self.strategies:
                orders = []
                # Determine what arguments to pass based on strategy type
                # MarketMakerStrategy expects an order book; others expect price only
                try:
                    # Use introspection to route calls; many strategies accept only price
                    if hasattr(strat, "generate_orders"):
                        # Market maker: pass synthetic order book
                        order_book = {
                            "bids": [(price, 1.0)],
                            "asks": [(price, 1.0)],
                        }
                        orders = strat.generate_orders(order_book)
                    elif hasattr(strat, "update"):
                        # Many strategies implement update method; pass price and empty trades list
                        # For MLStrategy we need trades; but we approximate with empty
                        if strat.__class__.__name__ == "MLStrategy":
                            orders = strat.update(price, [])
                        else:
                            orders = strat.update(price)
                except Exception:
                    # skip strategy on error
                    orders = []
                # Execute orders
                for side_order, order_price, order_qty in orders:
                    if side_order == "buy":
                        self.portfolio.buy(symbol, price if order_price is None else order_price, order_qty)
                        trades_executed.append((ts, "buy", order_qty, price if order_price is None else order_price))
                    elif side_order == "sell":
                        self.portfolio.sell(symbol, price if order_price is None else order_price, order_qty)
                        trades_executed.append((ts, "sell", order_qty, price if order_price is None else order_price))
            # Mark portfolio value
            current_value = self.portfolio.value(self.prices)
            self.pnl_history.append((ts, current_value))
        return {
            "portfolio": self.portfolio,
            "pnl": self.pnl_history,
            "trades": trades_executed,
        }
