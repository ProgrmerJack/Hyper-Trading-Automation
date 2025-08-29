"""Triple Barrier labeling (Lopez de Prado) with meta-label prep.
This is a pragmatic implementation for bar-based data.
"""
from typing import Tuple, Optional
import numpy as np
import pandas as pd

def get_daily_vol(close: pd.Series, span: int = 100) -> pd.Series:
    r = np.log(close).diff()
    return r.ewm(span=span).std().shift(1)

def apply_triple_barrier(close: pd.Series,
                         pt_mult: float = 2.0,
                         sl_mult: float = 2.0,
                         max_holding: int = 50,
                         vol: Optional[pd.Series] = None) -> pd.DataFrame:
    close = close.astype(float)
    if vol is None:
        vol = get_daily_vol(close)
    out = []
    for t0 in range(len(close)-1):
        if np.isnan(vol.iloc[t0]): 
            out.append((np.nan,np.nan,np.nan)); continue
        pt = close.iloc[t0]*(1+pt_mult*vol.iloc[t0])
        sl = close.iloc[t0]*(1-sl_mult*vol.iloc[t0])
        t1 = min(t0+max_holding, len(close)-1)
        path = close.iloc[t0+1:t1+1]
        hit_pt = (path>=pt).idxmax() if (path>=pt).any() else None
        hit_sl = (path<=sl).idxmax() if (path<=sl).any() else None
        if hit_pt is not None and (hit_sl is None or hit_pt<=hit_sl):
            label = 1; t_end = hit_pt
        elif hit_sl is not None:
            label = -1; t_end = hit_sl
        else:
            label = 0; t_end = close.index[t1]
        out.append((t_end, label, float((close.loc[t_end]-close.iloc[t0])/close.iloc[t0])))
    df = pd.DataFrame(out, index=close.index[:-1], columns=["t_end","label","ret"])
    return df

def meta_label(primary_pred: pd.Series, tb: pd.DataFrame) -> pd.Series:
    """Meta-label is 1 if primary prediction sign matches realized label (>0)."""
    aligned = primary_pred.reindex(tb.index).fillna(0.0)
    y = (aligned*np.sign(tb["label"]).replace(0, np.nan)).dropna()
    return (y>0).astype(int).reindex(tb.index).fillna(0).astype(int)
