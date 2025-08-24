from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
from .utils import mk_market
@dataclass
class SpotDCA:
    symbol: str; interval_sec: int; qty: float; max_trades: int = 0
    last_ts: int = 0; count: int = 0
    def on_clock(self, now_ms: int) -> List[Dict[str,Any]]:
        intents=[]
        if self.last_ts==0: self.last_ts=now_ms
        if now_ms-self.last_ts>=self.interval_sec*1000:
            self.last_ts=now_ms; self.count+=1
            intents.append(mk_market(self.symbol,'BUY',self.qty,bot='spot_dca'))
        return intents
