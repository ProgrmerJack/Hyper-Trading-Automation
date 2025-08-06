"""Simple backtesting harness."""

import pandas as pd
from pathlib import Path

from hypertrader.strategies.indicator_signals import generate_signal
from hypertrader.utils.performance import compute_returns, sharpe_ratio, max_drawdown


def run_backtest(data: pd.DataFrame) -> tuple[float, float, float]:

    balance = 10000.0
    position = None
    entry_price = 0.0

    for i in range(200, len(data)):
        window = data.iloc[: i + 1]
        signal = generate_signal(window)

        price = data["close"].iloc[i]
        if signal.action == "BUY" and position is None:
            position = "LONG"
            entry_price = price
        elif signal.action == "SELL" and position is not None:
            profit = price - entry_price
            balance += profit
            position = None
    if position is not None:
        balance += data["close"].iloc[-1] - entry_price
    returns = compute_returns(data["close"])
    sr = sharpe_ratio(returns)
    dd = max_drawdown(returns)
    return balance, sr, dd


def main():
    csv_path = Path("data/raw/sample.csv")
    if not csv_path.exists():
        print("No sample data for backtest")
        return
    df = pd.read_csv(csv_path, parse_dates=["timestamp"], index_col="timestamp")
    balance, sr, dd = run_backtest(df)
    print(f"Final balance: {balance:.2f} SR={sr:.2f} DD={dd:.2%}")


if __name__ == "__main__":
    main()
