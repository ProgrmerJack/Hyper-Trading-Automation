"""
Meta strategy for combining multiple trading strategies.

The ``MetaStrategy`` manages a collection of sub‑strategies and
aggregates their order recommendations.  This allows the bot to run
multiple strategies concurrently on the same or different
instruments.  The meta strategy can optionally filter or throttle
orders using a risk management callback or dynamic capital
allocation logic.  For example, one might allocate more capital to
market making during calm periods and shift to arbitrage when
mispricings appear.

The current implementation simply loops through its constituent
strategies, collects their orders and returns them.  Order sizes
should already be scaled appropriately within each strategy.  To
prioritise or merge conflicting orders, override the
``aggregate_orders`` method.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Callable, List, Tuple, Dict, Optional


@dataclass
class HedgeAllocator:
    """Simple hedge allocator for strategy weighting."""
    n: int
    eta: float = 0.1
    _weights: np.ndarray = field(init=False)
    
    def __post_init__(self):
        self._weights = np.ones(self.n) / self.n
    
    def weights(self) -> np.ndarray:
        return self._weights.copy()
    
    def update(self, proxy: List[float]) -> np.ndarray:
        """Update weights based on proxy performance."""
        proxy_arr = np.array(proxy)
        self._weights *= np.exp(self.eta * proxy_arr)
        self._weights /= self._weights.sum()
        return self._weights.copy()

@dataclass
class MetaStrategy:
    """Combine multiple strategies with adaptive weighting."""
    strategies: List[Any]
    risk_callback: Optional[Callable[[List[Tuple[str, Any, float]]], List[Tuple[str, Any, float]]]] = None
    eta: float = 0.1
    weights: np.ndarray = field(init=False)
    allocator: HedgeAllocator = field(init=False)
    
    def __post_init__(self):
        self.allocator = HedgeAllocator(n=len(self.strategies), eta=self.eta)
        self.weights = self.allocator.weights()

    def update(self, *args, **kwargs):
        """Update strategies with adaptive weighting or order aggregation."""
        # DataFrame-based update for weighted signals
        if len(args) == 1 and isinstance(args[0], pd.DataFrame):
            return self._update_weighted(args[0])
        # Traditional order aggregation
        return self._update_orders(*args, **kwargs)
    
    def _update_weighted(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Update with weighted signal aggregation."""
        sigs, confs = [], []
        for s in self.strategies:
            try:
                result = s.update(df)
                if isinstance(result, dict):
                    sig, conf = result.get('signal', 0), result.get('confidence', 0.5)
                elif isinstance(result, tuple) and len(result) >= 2:
                    sig, conf = result[0], result[1]
                else:
                    sig, conf = 0, 0.5
            except Exception:
                sig, conf = 0, 0.5
            sigs.append(sig)
            confs.append(conf)
        
        # Update weights based on recent performance
        last_ret = float(df['close'].pct_change().iloc[-1]) if len(df) >= 2 else 0.0
        proxy = [conf * (1 if (sig*last_ret) > 0 else -1 if (sig*last_ret) < 0 else 0) 
                for sig, conf in zip(sigs, confs)]
        self.weights = self.allocator.update(proxy)
        
        # Aggregate weighted signal
        agg_score = float(np.dot(self.weights, [s*c for s, c in zip(sigs, confs)]))
        sig = 1 if agg_score > 0.15 else -1 if agg_score < -0.15 else 0
        conf = float(min(1.0, max(0.0, abs(agg_score))))
        
        return {'signal': sig, 'confidence': conf, 'weights': self.weights.tolist(), 'proxies': proxy}
    
    def _update_orders(self, *args, **kwargs) -> List[Tuple[str, Any, float]]:
        """Traditional order aggregation method."""
        orders: List[Tuple[str, Any, float]] = []
        for strat in self.strategies:
            try:
                result = strat.update(*args, **kwargs)
                if result:
                    orders.extend(result)
            except Exception:
                continue
        if self.risk_callback is not None:
            orders = self.risk_callback(orders)
        return orders
