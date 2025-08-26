#!/usr/bin/env python3
"""Simple data fetching using CCXT directly."""

import ccxt
import pandas as pd
from datetime import datetime, timedelta

def fetch_binance_data(symbol="BTC/USDT", timeframe="1h", days=30):
    """Fetch data directly from Binance."""
    exchange = None
    try:
        exchange = ccxt.binance()
        
        # Calculate since timestamp
        since = exchange.milliseconds() - days * 24 * 60 * 60 * 1000
        
        # Fetch OHLCV data
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since)
        
        # Convert to DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        print(f"[SUCCESS] Fetched {len(df)} candles for {symbol}")
        return df
        
    except Exception as e:
        print(f"[ERROR] Failed to fetch data: {e}")
        return None
    finally:
        # Safely close the exchange if a close() method exists
        try:
            if exchange is not None:
                close_fn = getattr(exchange, "close", None)
                if callable(close_fn):
                    close_fn()
        except Exception:
            pass

if __name__ == "__main__":
    # Fetch BTC data
    data = fetch_binance_data("BTC/USDT", "1h", 30)
    if data is not None:
        data.to_csv("btc_real_data.csv")
        print("[SUCCESS] Data saved to btc_real_data.csv")
        print(f"Date range: {data.index[0]} to {data.index[-1]}")
        print(f"Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")