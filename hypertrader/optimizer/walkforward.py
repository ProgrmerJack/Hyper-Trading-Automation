from __future__ import annotations
import numpy as np, pandas as pd
from typing import List, Dict, Any, Callable, Tuple
from ..backtester import Backtester, Portfolio
from ..strategies import MetaStrategy, MLStrategy, DonchianBreakout, MeanReversionEMA, MomentumMultiTF
def walkforward_grid(df: pd.DataFrame, factory: Callable[[Dict[str,Any]], Any],
                     grid: Dict[str, List[Any]], window: int = 800, step: int = 200) -> List[Tuple[Dict[str,Any], float]]:
    keys = list(grid.keys()); from itertools import product
    combos = [dict(zip(keys, vals)) for vals in product(*grid.values())]
    results: List[Tuple[Dict[str,Any], float]] = []
    for start in range(0, max(1, len(df)-window), step):
        sub = df.iloc[start:start+window][['open','high','low','close','volume']]
        if len(sub) < window: break
        best_params, best_pnl = None, -1e9
        for params in combos:
            strat = factory(params)
            meta = MetaStrategy([strat, MLStrategy(), DonchianBreakout(), MeanReversionEMA(), MomentumMultiTF()])
            res = Backtester().run(sub, meta, Portfolio(cash=1000.0))
            if res['pnl'] > best_pnl: best_pnl, best_params = res['pnl'], params
        if best_params is not None: results.append((best_params, float(best_pnl)))
    return results
