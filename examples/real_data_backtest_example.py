#!/usr/bin/env python3
"""Complete example: Fetch real data and run backtests."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import ccxt
from datetime import datetime

# Import hypertrader modules
from hypertrader.strategies.indicator_signals import generate_signal
from hypertrader.utils.risk import calculate_position_size
from hypertrader.utils.performance import sharpe_ratio, max_drawdown

def fetch_real_data(symbol="BTC/USDT", timeframe="1h", days=30):
    """Fetch real market data."""
    print(f"Fetching {days} days of {symbol} data...")
    
    exchange = None
    try:
        exchange = ccxt.binance()
        since = exchange.milliseconds() - days * 24 * 60 * 60 * 1000
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since)
        
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        print(f"[SUCCESS] Fetched {len(df)} candles")
        return df
    finally:
        # Safely close the exchange if a close() method exists
        try:
            if exchange is not None:
                close_fn = getattr(exchange, "close", None)
                if callable(close_fn):
                    close_fn()
        except Exception:
            pass

def backtest_strategy(data, initial_balance=10000, risk_percent=2):
    """Backtest the hypertrader strategy."""
    print("Running backtest...")
    
    balance = initial_balance
    position = 0
    position_price = 0
    trades = []
    equity_curve = []
    
    for i in range(50, len(data)):  # Need history for indicators
        current_data = data.iloc[:i+1]
        price = current_data['close'].iloc[-1]
        
        # Generate signal using hypertrader strategy
        try:
            sig = generate_signal(current_data, 0, 0, 0, 0, 0)  # No sentiment/macro data
            action = sig.action
        except:
            action = "HOLD"
        
        # Execute trades
        if action == "BUY" and position <= 0:
            # Close short position if any
            if position < 0:
                pnl = abs(position) * (position_price - price)
                balance += pnl
                trades.append({"action": "close_short", "price": price, "pnl": pnl})
            
            # Open long position
            stop_loss = price * 0.98  # 2% stop loss
            pos_size = calculate_position_size(balance, risk_percent, price, stop_loss)
            if pos_size > 0:
                position = pos_size / price
                position_price = price
                trades.append({"action": "buy", "price": price, "size": position})
        
        elif action == "SELL" and position >= 0:
            # Close long position if any
            if position > 0:
                pnl = position * (price - position_price)
                balance += pnl
                trades.append({"action": "close_long", "price": price, "pnl": pnl})
            
            # Open short position
            stop_loss = price * 1.02  # 2% stop loss
            pos_size = calculate_position_size(balance, risk_percent, price, stop_loss)
            if pos_size > 0:
                position = -pos_size / price
                position_price = price
                trades.append({"action": "sell", "price": price, "size": abs(position)})
        
        # Calculate current equity
        if position > 0:
            current_equity = balance + position * price
        elif position < 0:
            current_equity = balance + abs(position) * (position_price - price)
        else:
            current_equity = balance
        
        equity_curve.append(current_equity)
    
    # Close final position
    if position != 0:
        final_price = data['close'].iloc[-1]
        if position > 0:
            pnl = position * (final_price - position_price)
        else:
            pnl = abs(position) * (position_price - final_price)
        balance += pnl
        trades.append({"action": "final_close", "price": final_price, "pnl": pnl})
    
    return balance, trades, equity_curve

def analyze_results(initial_balance, final_balance, trades, equity_curve):
    """Analyze backtest results."""
    total_return = (final_balance / initial_balance - 1) * 100
    
    # Calculate returns series
    returns = pd.Series(equity_curve).pct_change().dropna()
    
    # Performance metrics
    sharpe = sharpe_ratio(returns) if len(returns) > 1 else 0
    max_dd = max_drawdown(returns) * 100 if len(returns) > 1 else 0
    
    # Trade analysis
    profitable_trades = [t for t in trades if t.get('pnl', 0) > 0]
    losing_trades = [t for t in trades if t.get('pnl', 0) < 0]
    
    print(f"\n=== BACKTEST RESULTS ===")
    print(f"Initial Balance: ${initial_balance:,.2f}")
    print(f"Final Balance: ${final_balance:,.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {max_dd:.2f}%")
    print(f"Total Trades: {len(trades)}")
    
    if trades:
        pnl_trades = [t for t in trades if 'pnl' in t]
        if pnl_trades:
            print(f"Profitable Trades: {len(profitable_trades)}/{len(pnl_trades)} ({len(profitable_trades)/len(pnl_trades)*100:.1f}%)")
            
            if profitable_trades:
                avg_win = np.mean([t['pnl'] for t in profitable_trades])
                print(f"Average Win: ${avg_win:.2f}")
            
            if losing_trades:
                avg_loss = np.mean([t['pnl'] for t in losing_trades])
                print(f"Average Loss: ${avg_loss:.2f}")

def main():
    """Main execution function."""
    print("=== REAL DATA BACKTESTING EXAMPLE ===\n")
    
    # Step 1: Fetch real data
    try:
        data = fetch_real_data("BTC/USDT", "1h", 30)
        print(f"Data range: {data.index[0]} to {data.index[-1]}")
        print(f"Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}\n")
    except Exception as e:
        print(f"Error fetching data: {e}")
        return
    
    # Step 2: Run backtest
    try:
        final_balance, trades, equity_curve = backtest_strategy(data)
        
        # Step 3: Analyze results
        analyze_results(10000, final_balance, trades, equity_curve)
        
    except Exception as e:
        print(f"Backtest error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()