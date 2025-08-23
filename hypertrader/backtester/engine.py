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
import pandas as pd

from ..connectors.exchange import SimulationConnector


@dataclass
class FeeModel:
    maker_fee: float = 0.001
    taker_fee: float = 0.001
    
    def fee(self, side: str, qty: float, price: float, maker: bool) -> float:
        return qty * price * (self.maker_fee if maker else self.taker_fee)

@dataclass
class PnLLedger:
    position: float = 0.0
    fees_paid: float = 0.0
    
    def on_fill(self, side: str, qty: float, price: float, fee: float, maker: bool) -> None:
        self.position += qty if side == 'buy' else -qty
        self.fees_paid += fee
    
    def net(self, mark_price: float) -> float:
        return self.position * mark_price - self.fees_paid

@dataclass
class Portfolio:
    """Portfolio for backtesting with fee modeling and PnL tracking."""
    cash: float = 1000.0
    positions: Dict[str, float] = field(default_factory=dict)
    ledger: PnLLedger = field(default_factory=PnLLedger)

    def buy(self, symbol: str, price: float, qty: float, fee: float = 0.0) -> None:
        cost = price * qty + fee
        self.cash -= cost
        self.positions[symbol] = self.positions.get(symbol, 0.0) + qty
        self.ledger.on_fill('buy', qty, price, fee, True)

    def sell(self, symbol: str, price: float, qty: float, fee: float = 0.0) -> None:
        proceeds = price * qty - fee
        self.cash += proceeds
        self.positions[symbol] = self.positions.get(symbol, 0.0) - qty
        self.ledger.on_fill('sell', qty, price, fee, True)

    def value(self, prices: Dict[str, float]) -> float:
        val = self.cash
        for sym, qty in self.positions.items():
            price = prices.get(sym, 0.0)
            val += qty * price
        return val


class Backtester:
    """Run an event‑driven backtest over historical trade data."""

    def __init__(self, connector: SimulationConnector = None, strategies: List[Any] = None, 
                 start_cash: float = 10_000.0, fee: FeeModel = None, slippage_bps: float = 1.0):
        self.connector = connector
        self.strategies = strategies or []
        self.portfolio = Portfolio(cash=start_cash)
        self.fee = fee or FeeModel()
        self.slip = slippage_bps * 1e-4
        self.prices: Dict[str, float] = {}
        self.pnl_history: List[Tuple[_dt.datetime, float]] = []

    def run(self, symbol: str = None, df: pd.DataFrame = None, meta_strategy = None) -> Dict[str, Any]:
        """Execute backtest on symbol or DataFrame."""
        if df is not None and meta_strategy is not None:
            return self._run_dataframe(df, meta_strategy)
        return self._run_symbol(symbol)
    
    def _run_dataframe(self, df: pd.DataFrame, meta_strategy) -> Dict[str, Any]:
        """Run backtest on DataFrame with meta strategy."""
        orders: List[Dict[str,Any]] = []
        for i in range(120, len(df)):
            view = df.iloc[:i+1][['open','high','low','close','volume']]
            out = meta_strategy.update(view)
            sig = out['signal']; conf = out['confidence']
            price = float(view['close'].iloc[-1]) * (1 + (self.slip if sig>0 else -self.slip) if sig!=0 else 0)
            if sig != 0:
                qty = (self.portfolio.cash * (0.005 + 0.01*conf)) / price
                fee = self.fee.fee('buy' if sig>0 else 'sell', qty, price, True)
                if sig > 0:
                    self.portfolio.buy('symbol', price, qty, fee)
                else:
                    self.portfolio.sell('symbol', price, qty, fee)
                orders.append({'side': 'buy' if sig>0 else 'sell', 'qty': qty, 'price': price, 'conf': conf})
        mark = float(df['close'].iloc[-1])
        return {'orders': orders, 'pnl': self.portfolio.ledger.net(mark), 
                'fees': self.portfolio.ledger.fees_paid, 'pos': self.portfolio.ledger.position}
    
    def _run_symbol(self, symbol: str) -> Dict[str, Any]:
        """Run backtest on symbol using connector."""
        history = self.connector._data[symbol]
        trades_executed: List[Tuple[_dt.datetime, str, float, float]] = []
        for ts, price, qty, side in history:
            self.prices[symbol] = price
            for strat in self.strategies:
                orders = []
                try:
                    if hasattr(strat, "generate_orders"):
                        order_book = {"bids": [(price, 1.0)], "asks": [(price, 1.0)]}
                        orders = strat.generate_orders(order_book)
                    elif hasattr(strat, "update"):
                        orders = strat.update(price, []) if strat.__class__.__name__ == "MLStrategy" else strat.update(price)
                except Exception:
                    orders = []
                for side_order, order_price, order_qty in orders:
                    exec_price = price if order_price is None else order_price
                    fee = self.fee.fee(side_order, order_qty, exec_price, True)
                    if side_order == "buy":
                        self.portfolio.buy(symbol, exec_price, order_qty, fee)
                    elif side_order == "sell":
                        self.portfolio.sell(symbol, exec_price, order_qty, fee)
                    trades_executed.append((ts, side_order, order_qty, exec_price))
            current_value = self.portfolio.value(self.prices)
            self.pnl_history.append((ts, current_value))
        return {"portfolio": self.portfolio, "pnl": self.pnl_history, "trades": trades_executed}
