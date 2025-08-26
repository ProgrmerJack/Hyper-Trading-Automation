from __future__ import annotations
import pandas as pd
from dataclasses import dataclass
@dataclass
class DonchianBreakout:
    n:int = 55
    def update(self, df: pd.DataFrame):
        if len(df) < self.n+1: return 0, 0.5, {}
        hi = df['high'].rolling(self.n).max().iloc[-2]
        lo = df['low'].rolling(self.n).min().iloc[-2]
        c = df['close'].iloc[-1]
        if c > hi: return 1, 0.65, {'level': hi}
        if c < lo: return -1, 0.65, {'level': lo}
        return 0, 0.5, {}
