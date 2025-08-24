from __future__ import annotations
from dataclasses import dataclass
def mk_limit(symbol: str, side: str, price: float, qty: float, **tags):
    return {'symbol': symbol, 'side': side, 'type': 'LIMIT', 'price': float(price), 'qty': float(qty), 'tags': tags}
def mk_market(symbol: str, side: str, qty: float, **tags):
    return {'symbol': symbol, 'side': side, 'type': 'MARKET', 'qty': float(qty), 'tags': tags}
@dataclass
class Fill:
    side: str; price: float; qty: float; maker: bool = True
class Position:
    def __init__(self): self.qty=0.0; self.avg=0.0
    def on_fill(self, fill: Fill):
        s = +1 if fill.side.lower()=='buy' else -1
        new_qty = self.qty + s*fill.qty
        if self.qty==0 or (self.qty>0 and s>0) or (self.qty<0 and s<0):
            total = self.avg*abs(self.qty) + fill.price*fill.qty
            self.qty = new_qty; self.avg = total / max(abs(self.qty), 1e-12)
        else:
            closing = min(abs(fill.qty), abs(self.qty))
            self.qty = new_qty
            if self.qty==0: self.avg = 0.0
