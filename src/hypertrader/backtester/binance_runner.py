"""Backtest runner using cached Binance klines (CSV) and your strategies.
This avoids synthetic data. Fetch klines separately and store as CSV.

CSV schema expected: timestamp, open, high, low, close, volume (UTC ms or ISO ts).
"""
from pathlib import Path
import pandas as pd
import numpy as np

def load_klines_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()
    elif "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df = df.set_index("ts").sort_index()
    else:
        raise ValueError("CSV must have timestamp/ts column")
    return df[["open","high","low","close","volume"]].astype(float)

def simple_mom_strategy(df: pd.DataFrame, lookback: int = 50):
    ret = np.log(df["close"]).diff()
    sig = (df["close"] > df["close"].rolling(lookback).mean()).astype(int)
    pos = sig.shift(1).fillna(0)
    pnl = (pos * ret).fillna(0)
    return pnl

def evaluate(pnl: pd.Series):
    pnl = pnl.dropna()
    total_ret = float(np.exp(pnl.sum()) - 1.0)
    sharpe = float((pnl.mean()/ (pnl.std()+1e-12)) * np.sqrt(365*24*60))  # if minute bars
    dd = (pnl.cumsum() - pnl.cumsum().cummax()).min()
    return {"total_return": total_ret, "sharpe": sharpe, "max_drawdown": float(dd)}

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()
    df = load_klines_csv(Path(args.csv))
    pnl = simple_mom_strategy(df)
    print(evaluate(pnl))
