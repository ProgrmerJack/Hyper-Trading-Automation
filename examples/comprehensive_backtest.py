#!/usr/bin/env python3
"""Comprehensive backtest using the existing framework."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import pandas as pd
import numpy as np

def run_comprehensive_backtest():
    """Run comprehensive backtest using existing framework."""
    try:
        # Use the existing backtest framework
        from tests.backtest import run_backtest
        
        # Load real data
        data = pd.read_csv("data/btc_real_data.csv", index_col=0, parse_dates=True)
        print(f"[SUCCESS] Loaded {len(data)} candles")
        
        # Run backtest with different strategies
        strategies = ['enhanced', 'technical', 'comprehensive']
        
        for strategy in strategies:
            print(f"\n=== BACKTESTING {strategy.upper()} STRATEGY ===")
            try:
                result = run_backtest(strategy)
                if result:
                    print(f"Strategy: {strategy}")
                    print(f"Final Return: {result.get('total_return', 0):.2f}%")
                    print(f"Sharpe Ratio: {result.get('sharpe_ratio', 0):.2f}")
                    print(f"Max Drawdown: {result.get('max_drawdown', 0):.2f}%")
            except Exception as e:
                print(f"[ERROR] {strategy} backtest failed: {e}")
        
    except ImportError:
        print("[INFO] Using vectorbt for backtesting...")
        run_vectorbt_backtest()

def run_vectorbt_backtest():
    """Run backtest using vectorbt."""
    try:
        import vectorbt as vbt
        
        # Load data
        data = pd.read_csv("data/btc_real_data.csv", index_col=0, parse_dates=True)
        close = data['close']
        
        # Simple moving average crossover strategy
        fast_ma = close.rolling(10).mean()
        slow_ma = close.rolling(30).mean()
        
        # Generate signals
        entries = fast_ma > slow_ma
        exits = fast_ma < slow_ma
        
        # Run backtest
        pf = vbt.Portfolio.from_signals(close, entries, exits, init_cash=10000)
        
        print(f"\n=== VECTORBT BACKTEST RESULTS ===")
        print(f"Initial Cash: $10,000")
        print(f"Final Value: ${pf.value().iloc[-1]:,.2f}")
        print(f"Total Return: {pf.total_return() * 100:.2f}%")
        print(f"Sharpe Ratio: {pf.sharpe_ratio():.2f}")
        print(f"Max Drawdown: {pf.max_drawdown() * 100:.2f}%")
        print(f"Number of Trades: {pf.orders.count()}")
        
        return pf
        
    except Exception as e:
        print(f"[ERROR] VectorBT backtest failed: {e}")
        return None

def simple_ma_backtest():
    """Simple moving average backtest."""
    data = pd.read_csv("data/btc_real_data.csv", index_col=0, parse_dates=True)
    
    # Calculate indicators
    data['ma_fast'] = data['close'].rolling(10).mean()
    data['ma_slow'] = data['close'].rolling(30).mean()
    data['signal'] = 0
    data.loc[data['ma_fast'] > data['ma_slow'], 'signal'] = 1
    data.loc[data['ma_fast'] < data['ma_slow'], 'signal'] = -1
    
    # Calculate returns
    data['returns'] = data['close'].pct_change()
    data['strategy_returns'] = data['signal'].shift(1) * data['returns']
    
    # Performance metrics
    total_return = (1 + data['strategy_returns']).prod() - 1
    sharpe = data['strategy_returns'].mean() / data['strategy_returns'].std() * np.sqrt(24*365)  # Hourly to annual
    max_dd = (data['strategy_returns'].cumsum() - data['strategy_returns'].cumsum().expanding().max()).min()
    
    print(f"\n=== SIMPLE MA CROSSOVER BACKTEST ===")
    print(f"Total Return: {total_return * 100:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {max_dd * 100:.2f}%")
    
    return data

if __name__ == "__main__":
    # Check if data exists
    if not Path("btc_real_data.csv").exists():
        print("[ERROR] No data file found. Run simple_data_fetch.py first.")
        exit(1)
    
    # Run simple backtest first
    simple_ma_backtest()
    
    # Try comprehensive backtest
    run_comprehensive_backtest()