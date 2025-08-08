from __future__ import annotations

from typing import Iterable


class EventDrivenBacktester:
    """Very small event-driven simulator with fees and slippage."""

    def __init__(self, equity: float, fee_rate: float = 0.0005, slippage: float = 0.0005) -> None:
        self.equity = equity
        self.position = 0.0
        self.entry_price = 0.0
        self.fee_rate = fee_rate
        self.slippage = slippage

    def trade(self, side: str, qty: float, price: float) -> None:
        fill_price = price * (1 + self.slippage if side == "BUY" else 1 - self.slippage)
        cost = fill_price * qty
        fee = abs(cost) * self.fee_rate
        if side == "BUY":
            self.position += qty
            self.entry_price = fill_price
            self.equity -= cost + fee
        else:
            pnl = (fill_price - self.entry_price) * qty
            self.equity += pnl - fee
            self.position -= qty

    def process(self, prices: Iterable[float], signals: Iterable[str], qty: float) -> float:
        last_price = None
        for price, sig in zip(prices, signals):
            last_price = price
            if sig in ("BUY", "SELL"):
                self.trade(sig, qty, price)
        if self.position and last_price is not None:
            self.equity += (last_price - self.entry_price) * self.position
        return self.equity
