from __future__ import annotations
import numpy as np, pandas as pd
from dataclasses import dataclass

@dataclass
class MomentumMultiTF:
    fast:int = 12; mid:int = 48; slow:int = 96
    def update(self, df: pd.DataFrame):
        if len(df) < self.slow+2: return 0, 0.5, {}
        ema_f = df['close'].ewm(span=self.fast, adjust=False).mean().iloc[-1]
        ema_m = df['close'].ewm(span=self.mid, adjust=False).mean().iloc[-1]
        ema_s = df['close'].ewm(span=self.slow, adjust=False).mean().iloc[-1]
        c = df['close'].iloc[-1]
        score = (c>ema_f) + (c>ema_m) + (c>ema_s) - ((c<ema_f) + (c<ema_m) + (c<ema_s))
        if score>=2: return 1, 0.7, {'score': int(score)}
        if score<=-2: return -1, 0.7, {'score': int(score)}
        return 0, 0.5, {'score': int(score)}
