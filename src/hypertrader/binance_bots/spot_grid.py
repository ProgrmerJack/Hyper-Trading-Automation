from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
from .utils import mk_limit
@dataclass
class SpotGrid:
    symbol: str; lower: float; upper: float; grids: int; base_qty: float
    take_profit_pct: float = 0.0; rebalance_on_fill: bool = True
    def __post_init__(self):
        assert self.upper>self.lower and self.grids>=2
        step = (self.upper-self.lower)/(self.grids-1)
        self.levels = [self.lower+i*step for i in range(self.grids)]
        mid = 0.5*(self.lower+self.upper)
        self.sides = ['BUY' if l<mid else 'SELL' for l in self.levels]
        self.live = {}
    def bootstrap(self) -> List[Dict[str,Any]]:
        return [mk_limit(self.symbol,s,price=l,qty=self.base_qty,bot='spot_grid',level=l) for l,s in zip(self.levels,self.sides)]
    def on_tick(self, price: float) -> List[Dict[str,Any]]:
        intents=[]; 
        for l,s in zip(self.levels,self.sides):
            k=(l,s)
            if k not in self.live:
                intents.append(mk_limit(self.symbol,s,price=l,qty=self.base_qty,bot='spot_grid',level=l))
                self.live[k]=True
        return intents
    def on_fill(self, price: float, side: str, qty: float) -> List[Dict[str,Any]]:
        intents=[]; 
        try: idx=self.levels.index(price)
        except ValueError: return intents
        if side=='BUY' and idx+1<len(self.levels):
            intents.append(mk_limit(self.symbol,'SELL',price=self.levels[idx+1],qty=qty,bot='spot_grid',paired=True))
        elif side=='SELL' and idx-1>=0:
            intents.append(mk_limit(self.symbol,'BUY',price=self.levels[idx-1],qty=qty,bot='spot_grid',paired=True))
        return intents
