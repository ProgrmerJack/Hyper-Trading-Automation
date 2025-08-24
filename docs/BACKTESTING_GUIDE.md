# Real Data Backtesting Guide

This guide shows you how to acquire real market data and run backtests with the Hyper-Trading-Automation system.

## 🚀 Quick Start

### 1. Fetch Real Market Data

```bash
# Simple approach - fetch BTC data directly
conda activate trading-py310
python simple_data_fetch.py
```

This will:
- Fetch 30 days of BTC/USDT hourly data from Binance
- Save it to `btc_real_data.csv`
- Show data range and price statistics

### 2. Run Backtests

#### Option A: Simple Moving Average Strategy
```bash
python comprehensive_backtest.py
```

#### Option B: Hypertrader Strategy
```bash
python real_data_backtest_example.py
```

#### Option C: VectorBT Framework
```bash
python comprehensive_backtest.py  # Uses vectorbt automatically
```

## 📊 Available Backtesting Methods

### 1. **Simple Data Fetching** (`simple_data_fetch.py`)
- Direct CCXT integration
- Fetches OHLCV data from Binance
- Saves to CSV for reuse

### 2. **VectorBT Backtesting** (`comprehensive_backtest.py`)
- Professional backtesting framework
- Fast vectorized calculations
- Comprehensive performance metrics
- Multiple strategy support

### 3. **Custom Strategy Backtesting** (`real_data_backtest_example.py`)
- Uses hypertrader's signal generation
- Includes risk management
- Position sizing with stop losses
- Detailed trade analysis

## 📈 Sample Results

From our test run with 30 days of BTC data:

```
=== VECTORBT BACKTEST RESULTS ===
Initial Cash: $10,000
Final Value: $9,690.03
Total Return: -3.10%
Sharpe Ratio: -2.41
Max Drawdown: -5.45%
Number of Trades: 18
```

## 🔧 Customization Options

### Fetch Different Assets
```python
# In simple_data_fetch.py, modify:
data = fetch_binance_data("ETH/USDT", "1h", 30)  # Ethereum
data = fetch_binance_data("BTC/USDT", "4h", 90)  # 4-hour candles, 90 days
```

### Different Timeframes
- `"1m"` - 1 minute
- `"5m"` - 5 minutes  
- `"1h"` - 1 hour
- `"4h"` - 4 hours
- `"1d"` - 1 day

### Strategy Parameters
```python
# In backtest scripts, modify:
initial_balance = 50000  # Starting capital
risk_percent = 1.5       # Risk per trade
```

## 🎯 Advanced Features

### Multi-Strategy Backtesting
The system supports backtesting multiple strategies simultaneously:
- Technical indicators (MA, RSI, Bollinger Bands, etc.)
- Machine learning models
- Binance-style bots (Grid, DCA, Arbitrage)
- Risk management overlays

### Performance Metrics
- Total return
- Sharpe ratio
- Maximum drawdown
- Win/loss ratio
- Average win/loss amounts
- Number of trades

## 📝 Files Created

After running the examples, you'll have:
- `btc_real_data.csv` - Raw market data
- Performance reports in console output
- Optional: Trade logs and equity curves

## 🚨 Important Notes

1. **No Live Trading**: All backtests are simulation only
2. **Historical Data**: Results don't guarantee future performance  
3. **Slippage/Fees**: Not included in simple backtests
4. **Market Conditions**: 30-day sample may not represent all market regimes

## 🔄 Next Steps

1. **Extend Data Range**: Fetch more historical data for robust testing
2. **Add More Assets**: Test across different cryptocurrencies
3. **Optimize Parameters**: Use the built-in optimization tools
4. **Paper Trading**: Test strategies in real-time without risk
5. **Live Trading**: Deploy with proper risk management (when ready)

## 📚 Additional Resources

- `tests/backtest.py` - Built-in backtest framework
- `run_backtests.py` - Comprehensive strategy testing
- `hypertrader/strategies/` - Available trading strategies
- `hypertrader/utils/performance.py` - Performance calculation utilities