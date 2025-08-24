# Scripts Directory

This directory contains utility scripts for data fetching and system operations.

## Data Fetching Scripts

### `simple_data_fetch.py`
- Fetches real market data from Binance using CCXT
- Saves OHLCV data to CSV files
- Usage: `python simple_data_fetch.py`

### `get_real_data.py` 
- Alternative data fetching using hypertrader's fetch_data module
- More integrated with the hypertrader system
- Usage: `python get_real_data.py`

## Usage

```bash
# Activate the trading environment
conda activate trading-py310

# Fetch 30 days of BTC data
cd scripts
python simple_data_fetch.py

# Data will be saved to ../data/
```

## Configuration

Modify the scripts to change:
- Symbol: `"BTC/USDT"`, `"ETH/USDT"`, etc.
- Timeframe: `"1m"`, `"1h"`, `"4h"`, `"1d"`
- Duration: Number of days to fetch