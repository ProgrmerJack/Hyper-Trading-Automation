from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List
@dataclass
class HedgeAllocator:
    n: int; eta: float = 0.1; floor: float = 1e-6
    w: np.ndarray = field(init=False)
    def __post_init__(self): self.w = np.ones(self.n) / self.n
    def update(self, returns: List[float]):
        r = np.array(returns, dtype=float)
        self.w *= np.exp(self.eta * r); self.w = np.maximum(self.w, self.floor); self.w /= np.sum(self.w)
        return self.w.copy()
    def weights(self): return self.w.copy()
