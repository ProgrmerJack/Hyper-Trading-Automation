# Examples Directory

This directory contains example implementations and backtesting scripts.

## Backtesting Examples

### `comprehensive_backtest.py`
- Multiple backtesting approaches (VectorBT, Simple MA)
- Performance metrics calculation
- Usage: `python comprehensive_backtest.py`

### `real_data_backtest_example.py`
- Complete example using hypertrader strategies
- Fetches real data and runs backtest
- Includes risk management and performance analysis
- Usage: `python real_data_backtest_example.py`

### `run_backtest.py`
- Simple backtest implementation
- Uses pre-saved CSV data
- Basic strategy testing
- Usage: `python run_backtest.py`

## Prerequisites

1. **Environment Setup**:
   ```bash
   conda activate trading-py310
   ```

2. **Data Requirements**:
   - Run data fetching scripts first
   - Or ensure CSV data files exist in `../data/`

## Example Workflow

```bash
# 1. Fetch data
cd ../scripts
python simple_data_fetch.py

# 2. Run backtest
cd ../examples  
python real_data_backtest_example.py

# 3. View results
# Results displayed in console with performance metrics
```

## Customization

Each example can be modified for:
- Different trading symbols
- Various timeframes
- Custom strategy parameters
- Risk management settings