from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
@dataclass
class Fill:
    side: str; qty: float; price: float; fee: float; maker: bool
@dataclass
class PnLLedger:
    fills: List[Fill] = field(default_factory=list)
    realized: float = 0.0
    fees_paid: float = 0.0
    position: float = 0.0
    avg_price: float = 0.0
    def on_fill(self, side: str, qty: float, price: float, fee: float, maker: bool):
        if qty <= 0: return
        self.fills.append(Fill(side, qty, price, fee, maker))
        self.fees_paid += fee
        sign = 1.0 if side == 'buy' else -1.0
        new_pos = self.position + sign * qty
        if self.position == 0 or (self.position > 0 and sign > 0) or (self.position < 0 and sign < 0):
            total_cost = self.avg_price * abs(self.position) + price * qty
            self.position = new_pos; self.avg_price = total_cost / max(abs(self.position), 1e-12)
        else:
            closing = min(abs(qty), abs(self.position))
            pnl = (price - self.avg_price) * (closing if self.position > 0 else -closing)
            self.realized += pnl; self.position = new_pos
            if self.position == 0: self.avg_price = 0.0
    def unrealized(self, mark_price: float) -> float:
        return (mark_price - self.avg_price) * self.position
    def net(self, mark_price: float) -> float:
        return self.realized + self.unrealized(mark_price) - self.fees_paid
