from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Any
from .utils import mk_market
@dataclass
class RebalancingBot:
    targets: Dict[str,float]; threshold: float = 0.02; total_equity: float = 1000.0
    def decide(self, prices: Dict[str,float], holdings: Dict[str,float]) -> List[Dict[str,Any]]:
        values={sym:holdings.get(sym,0.0)*prices[sym] for sym in self.targets}; eq=sum(values.values()) or self.total_equity
        intents=[]
        for sym,tgt in self.targets.items():
            cur=values.get(sym,0.0)/eq; delta=tgt-cur
            if abs(delta)>=self.threshold:
                notional=delta*eq; side='BUY' if notional>0 else 'SELL'; qty=abs(notional)/prices[sym]
                intents.append(mk_market(sym,side,qty,bot='rebalancing',delta=delta))
        return intents
