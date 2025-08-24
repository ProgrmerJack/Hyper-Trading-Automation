#!/usr/bin/env python3
"""Fetch real market data for backtesting."""

import pandas as pd
from hypertrader.data.fetch_data import fetch_ohlcv

def get_real_data(symbol="BTC/USDT", timeframe="1h", days=30):
    """Fetch real OHLCV data from Binance."""
    try:
        data = fetch_ohlcv("binance", symbol, timeframe, limit=days*24 if timeframe=="1h" else days*1440)
        print(f"[SUCCESS] Fetched {len(data)} candles for {symbol}")
        return data
    except Exception as e:
        print(f"[ERROR] Error fetching data: {e}")
        return None

if __name__ == "__main__":
    # Fetch data and save to CSV
    data = get_real_data("BTC/USDT", "1h", 30)
    if data is not None:
        data.to_csv("btc_data.csv")
        print("[SUCCESS] Data saved to btc_data.csv")