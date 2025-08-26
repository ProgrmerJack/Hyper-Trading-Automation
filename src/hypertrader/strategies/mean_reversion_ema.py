from __future__ import annotations
import numpy as np, pandas as pd
from dataclasses import dataclass
@dataclass
class MeanReversionEMA:
    n:int = 50; z_entry: float = 1.5; z_exit: float = 0.5
    def update(self, df: pd.DataFrame):
        if len(df) < self.n+5: return 0, 0.5, {}
        ema = df['close'].ewm(span=self.n, adjust=False).mean()
        spread = df['close'] - ema
        mu = spread.rolling(self.n).mean().iloc[-1]
        sd = spread.rolling(self.n).std().iloc[-1] + 1e-9
        z = (spread.iloc[-1]-mu)/sd
        if z > self.z_entry: return -1, 0.6, {'z': float(z)}
        if z < -self.z_entry: return 1, 0.6, {'z': float(z)}
        if abs(z) < self.z_exit: return 0, 0.5, {'z': float(z)}
        return 0, 0.55, {'z': float(z)}
