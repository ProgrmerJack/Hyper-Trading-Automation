from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
from .utils import mk_limit
@dataclass
class FuturesGrid:
    symbol: str; lower: float; upper: float; grids: int; base_qty: float
    hedge_mode: bool = True
    def __post_init__(self):
        assert self.upper>self.lower and self.grids>=2
        step=(self.upper-self.lower)/(self.grids-1)
        self.levels=[self.lower+i*step for i in range(self.grids)]
        mid=0.5*(self.lower+self.upper)
        self.sides=['BUY' if l<mid else 'SELL' for l in self.levels]
    def bootstrap(self)->List[Dict[str,Any]]:
        return [mk_limit(self.symbol,s,price=l,qty=self.base_qty,bot='futures_grid',level=l) for l,s in zip(self.levels,self.sides)]
    def on_fill(self, price: float, side: str, qty: float)->List[Dict[str,Any]]:
        intents=[]; 
        if side=='BUY':
            intents.append({'symbol':self.symbol,'side':'SELL','type':'LIMIT','price':price*1.002,'qty':qty,'reduceOnly':True,'tags':{'bot':'futures_grid','paired':True}})
        elif side=='SELL':
            intents.append({'symbol':self.symbol,'side':'BUY','type':'LIMIT','price':price*0.998,'qty':qty,'reduceOnly':True,'tags':{'bot':'futures_grid','paired':True}})
        return intents
