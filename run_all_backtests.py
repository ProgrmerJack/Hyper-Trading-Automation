#!/usr/bin/env python3
"""Run all three backtesting approaches as requested."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

import pandas as pd
import numpy as np

def approach_1_comprehensive_framework():
    """Approach 1: Run our comprehensive backtest framework."""
    print("\n" + "="*60)
    print("APPROACH 1: COMPREHENSIVE FRAMEWORK")
    print("="*60)
    
    try:
        from scripts.comprehensive_backtest import main
        import asyncio
        asyncio.run(main())
        return True
    except Exception as e:
        print(f"[ERROR] Comprehensive framework failed: {e}")
        return False

def approach_2_vectorbt():
    """Approach 2: VectorBT backtesting."""
    print("\n" + "="*60)
    print("APPROACH 2: VECTORBT BACKTESTING")
    print("="*60)
    
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
        
        # Run backtest with $1000 initial capital (medium risk)
        pf = vbt.Portfolio.from_signals(close, entries, exits, init_cash=1000)
        
        print(f"Initial Cash: $1,000")
        print(f"Final Value: ${pf.value().iloc[-1]:,.2f}")
        print(f"Total Return: {pf.total_return() * 100:.2f}%")
        print(f"Sharpe Ratio: {pf.sharpe_ratio():.2f}")
        print(f"Max Drawdown: {pf.max_drawdown() * 100:.2f}%")
        print(f"Number of Trades: {pf.orders.count()}")
        
        return pf
        
    except ImportError:
        print("[ERROR] VectorBT not installed. Install with: pip install vectorbt")
        return None
    except Exception as e:
        print(f"[ERROR] VectorBT backtest failed: {e}")
        return None

def approach_3_simple_pandas():
    """Approach 3: Simple pandas-based backtest."""
    print("\n" + "="*60)
    print("APPROACH 3: SIMPLE PANDAS BACKTESTING")
    print("="*60)
    
    try:
        data = pd.read_csv("data/btc_real_data.csv", index_col=0, parse_dates=True)
        
        # Calculate indicators
        data['ma_fast'] = data['close'].rolling(10).mean()
        data['ma_slow'] = data['close'].rolling(30).mean()
        data['signal'] = 0
        data.loc[data['ma_fast'] > data['ma_slow'], 'signal'] = 1
        data.loc[data['ma_fast'] < data['ma_slow'], 'signal'] = -1
        
        # Medium risk position sizing (50% allocation vs 100%)
        data['position'] = data['signal'] * 0.5  # 50% allocation for medium risk
        
        # Calculate returns
        data['returns'] = data['close'].pct_change()
        data['strategy_returns'] = data['position'].shift(1) * data['returns']
        
        # Performance metrics for $1000 starting capital
        initial_capital = 1000
        data['equity_curve'] = initial_capital * (1 + data['strategy_returns']).cumprod()
        
        total_return = (data['equity_curve'].iloc[-1] / initial_capital - 1) * 100
        sharpe = data['strategy_returns'].mean() / data['strategy_returns'].std() * np.sqrt(24*365)  # Hourly to annual
        
        # Max drawdown calculation
        rolling_max = data['equity_curve'].expanding().max()
        drawdowns = (data['equity_curve'] - rolling_max) / rolling_max
        max_drawdown = drawdowns.min() * 100
        
        # Count trades (signal changes)
        trades = (data['signal'].diff() != 0).sum()
        
        print(f"Initial Capital: ${initial_capital:,}")
        print(f"Final Value: ${data['equity_curve'].iloc[-1]:,.2f}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2f}%")
        print(f"Number of Trades: {trades}")
        
        return data
        
    except Exception as e:
        print(f"[ERROR] Simple backtest failed: {e}")
        return None

def main():
    """Run all three backtesting approaches."""
    print("RUNNING ALL THREE BACKTESTING APPROACHES")
    print("Medium Risk Management for $1000 Target")
    
    # Approach 1: Comprehensive Framework
    approach_1_comprehensive_framework()
    
    # Approach 2: VectorBT
    approach_2_vectorbt() 
    
    # Approach 3: Simple Pandas
    approach_3_simple_pandas()
    
    print("\n" + "="*60)
    print("ALL BACKTESTING APPROACHES COMPLETED")
    print("="*60)

if __name__ == "__main__":
    main()
