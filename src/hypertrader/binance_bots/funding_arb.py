from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
from .utils import mk_market
@dataclass
class FundingArbBot:
    symbol: str; base_qty: float; expected_apr_min: float = 0.05; hedge_rebalance_thresh: float = 0.01
    def estimate_apr(self, funding_rate_8h: float) -> float: return funding_rate_8h*3*365
    def decide(self, spot_px: float, mark_px: float, funding_rate_8h: float) -> List[Dict[str,Any]]:
        apr=self.estimate_apr(funding_rate_8h); intents=[]
        if apr>=self.expected_apr_min and funding_rate_8h>0:
            intents.append(mk_market(self.symbol,'BUY',self.base_qty,bot='funding_arb',leg='spot_long'))
            intents.append({'symbol':self.symbol,'side':'SELL','type':'MARKET','qty':self.base_qty,'tags':{'bot':'funding_arb','leg':'perp_short'}})
        elif apr>=self.expected_apr_min and funding_rate_8h<0:
            intents.append(mk_market(self.symbol,'SELL',self.base_qty,bot='funding_arb',leg='spot_short'))
            intents.append({'symbol':self.symbol,'side':'BUY','type':'MARKET','qty':self.base_qty,'tags':{'bot':'funding_arb','leg':'perp_long'}})
        return intents
    def rebalance(self, spot_pos_qty: float, perp_pos_qty: float) -> List[Dict[str,Any]]:
        diff=spot_pos_qty+perp_pos_qty; intents=[]
        if abs(diff)>self.hedge_rebalance_thresh*max(1e-9,abs(spot_pos_qty)):
            side='SELL' if diff>0 else 'BUY'; intents.append(mk_market(self.symbol,side,abs(diff),bot='funding_arb',action='rebalance'))
        return intents
