import pandas as pd
from hypertrader.utils.performance import compute_returns, sharpe_ratio, max_drawdown


def test_performance_metrics():
    prices = pd.Series([1, 1.1, 1.2, 1.1, 1.3])
    returns = compute_returns(prices)
    sr = sharpe_ratio(returns)
    dd = max_drawdown(returns)
    assert len(returns) == 4
    assert sr != 0
    assert dd <= 0
