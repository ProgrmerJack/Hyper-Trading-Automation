import pandas as pd

from hypertrader.backtest import advanced_backtest


def ma_strategy(data: pd.DataFrame):
    fast = data["close"].rolling(2).mean()
    slow = data["close"].rolling(3).mean()
    entries = fast > slow
    exits = fast < slow
    return entries, exits


def test_advanced_backtest():
    prices = pd.Series([1, 2, 3, 4, 5, 4, 3, 4, 5, 6], name="close")
    data = pd.DataFrame({"close": prices})
    stats1 = advanced_backtest(data, ma_strategy, slippage_bps=0.0, fee_bps=0.0, leverage=1)
    stats2 = advanced_backtest(data, ma_strategy, slippage_bps=0.0, fee_bps=0.0, leverage=2)
    assert stats1["final_value"] > 0
    assert "sharpe_ratio" in stats1
    assert stats2["final_value"] > stats1["final_value"]
    stats3 = advanced_backtest(data, ma_strategy, slippage_bps=100.0, fee_bps=100.0)
    assert stats3["final_value"] < stats1["final_value"]
