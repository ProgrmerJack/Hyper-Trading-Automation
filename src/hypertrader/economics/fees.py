from __future__ import annotations
from dataclasses import dataclass
@dataclass
class FeeModel:
    maker: float = 0.0002
    taker: float = 0.0004
    def fee(self, side: str, qty: float, price: float, maker_fill: bool) -> float:
        rate = self.maker if maker_fill else self.taker
        return rate * qty * price
