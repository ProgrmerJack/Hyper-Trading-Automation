# Data Directory

This directory contains market data, backtest results, and database files.

## Structure

```
data/
├── raw/                    # Raw market data files
│   └── sample.csv         # Sample data
├── backtest_results/       # Backtest output files
│   ├── backtest_results.json
│   └── comprehensive_backtest_results.json
├── btc_real_data.csv      # Real BTC market data
├── state.db               # Bot state database
└── README.md              # This file
```

## Data Files

### Market Data
- **`btc_real_data.csv`**: Real BTC/USDT hourly data from Binance
- **`raw/sample.csv`**: Sample data for testing

### Results
- **`backtest_results/`**: JSON files containing backtest performance metrics
- **`state.db`**: SQLite database storing bot state and order history

## Usage

### Loading Data
```python
import pandas as pd

# Load BTC data
data = pd.read_csv('data/btc_real_data.csv', index_col=0, parse_dates=True)

# Load backtest results
import json
with open('data/backtest_results/backtest_results.json') as f:
    results = json.load(f)
```

### Data Sources
- **Binance**: Primary source for cryptocurrency data
- **CCXT**: Library used for data fetching
- **Real-time**: WebSocket feeds for live data

## Maintenance

- Data files are automatically created by scripts
- Old data can be archived or deleted as needed
- Database files should be backed up regularly