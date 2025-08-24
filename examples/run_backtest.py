#!/usr/bin/env python3
"""Run backtest with real data."""

import pandas as pd
import numpy as np
from hypertrader.strategies.indicator_signals import generate_signal
from hypertrader.utils.risk import calculate_position_size
from hypertrader.bot import initialize_all_strategies, generate_all_strategy_signals

def simple_backtest(data, initial_balance=10000, risk_percent=2):
    """Run simple backtest on real data."""
    balance = initial_balance
    position = 0
    trades = []
    
    print(f"Starting backtest with {len(data)} candles...")
    
    for i in range(50, len(data)):  # Start after 50 candles for indicators
        current_data = data.iloc[:i+1]
        price = current_data['close'].iloc[-1]
        
        # Generate signal
        try:
            sig = generate_signal(current_data, 0, 0, 0, 0, 0)
            action = sig.action
        except:
            action = "HOLD"
        
        # Execute trades
        if action == "BUY" and position <= 0:
            if position < 0:  # Close short
                pnl = position * (position_price - price)
                balance += pnl
                trades.append({"type": "close_short", "price": price, "pnl": pnl})
            
            # Open long
            position_size = calculate_position_size(balance, risk_percent, price, price * 0.98)
            position = position_size / price
            position_price = price
            trades.append({"type": "buy", "price": price, "size": position})
            
        elif action == "SELL" and position >= 0:
            if position > 0:  # Close long
                pnl = position * (price - position_price)
                balance += pnl
                trades.append({"type": "close_long", "price": price, "pnl": pnl})
            
            # Open short
            position_size = calculate_position_size(balance, risk_percent, price, price * 1.02)
            position = -position_size / price
            position_price = price
            trades.append({"type": "sell", "price": price, "size": abs(position)})
    
    # Close final position
    if position != 0:
        final_price = data['close'].iloc[-1]
        if position > 0:
            pnl = position * (final_price - position_price)
        else:
            pnl = position * (position_price - final_price)
        balance += pnl
        trades.append({"type": "final_close", "price": final_price, "pnl": pnl})
    
    return balance, trades

def run_backtest_from_file(filename="btc_real_data.csv"):
    """Run backtest from saved data file."""
    try:
        data = pd.read_csv(filename, index_col=0, parse_dates=True)
        print(f"[SUCCESS] Loaded {len(data)} candles from {filename}")
        
        final_balance, trades = simple_backtest(data)
        
        print(f"\n=== BACKTEST RESULTS ===")
        print(f"Initial Balance: $10,000")
        print(f"Final Balance: ${final_balance:,.2f}")
        print(f"Total Return: {((final_balance/10000)-1)*100:.2f}%")
        print(f"Number of Trades: {len(trades)}")
        
        if trades:
            profitable_trades = [t for t in trades if t.get('pnl', 0) > 0]
            print(f"Profitable Trades: {len(profitable_trades)}/{len([t for t in trades if 'pnl' in t])}")
        
        return final_balance, trades
        
    except FileNotFoundError:
        print(f"[ERROR] File {filename} not found. Run simple_data_fetch.py first.")
        return None, None

if __name__ == "__main__":
    run_backtest_from_file()