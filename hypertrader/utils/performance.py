import numpy as np
import pandas as pd


def compute_returns(prices: pd.Series) -> pd.Series:
    """Compute simple returns from price series."""
    return prices.pct_change().dropna()


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualized Sharpe ratio using daily returns."""
    if returns.empty:
        return 0.0
    excess = returns - risk_free_rate / len(returns)
    return np.sqrt(len(returns)) * excess.mean() / excess.std(ddof=0)


def max_drawdown(returns: pd.Series) -> float:
    """Compute maximum drawdown of a return series."""
    if returns.empty:
        return 0.0
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    drawdown = cum / peak - 1
    return drawdown.min()
