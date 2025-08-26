# Project Structure

## Overview
The Hyper-Trading-Automation project is now organized into a clean, modular structure with proper separation of concerns.

## Directory Structure

```
Hyper-Trading-Automation/
├── 📁 hypertrader/           # Main package
│   ├── 📁 strategies/        # Trading strategies & signals
│   ├── 📁 binance_bots/     # Binance-style trading bots
│   ├── 📁 data/             # Data fetching & storage
│   ├── 📁 risk/             # Risk management
│   ├── 📁 execution/        # Order execution & management
│   ├── 📁 utils/            # Utilities & indicators
│   ├── 📁 backtester/       # Backtesting engines
│   ├── 📁 feeds/            # WebSocket data feeds
│   ├── 📁 indicators/       # Technical indicators
│   └── 📄 bot.py            # Main trading bot
│
├── 📁 scripts/              # Utility scripts
│   ├── 📄 simple_data_fetch.py    # Fetch market data
│   └── 📄 get_real_data.py        # Alternative data fetcher
│
├── 📁 examples/             # Example implementations
│   ├── 📄 real_data_backtest_example.py  # Complete backtest example
│   ├── 📄 comprehensive_backtest.py      # Multiple strategies
│   └── 📄 run_backtest.py               # Simple backtest
│
├── 📁 data/                 # Data storage
│   ├── 📁 raw/             # Raw market data
│   ├── 📁 backtest_results/ # Backtest outputs
│   ├── 📄 btc_real_data.csv # Sample BTC data
│   └── 📄 state.db          # Bot state database
│
├── 📁 docs/                 # Documentation
│   ├── 📄 BACKTESTING_GUIDE.md     # Backtesting guide
│   ├── 📄 production_readiness.md  # Production notes
│   └── 📄 *.pdf                    # Design documents
│
└── 📁 tests/                # Test suite
    ├── 📄 test_*.py         # Unit tests
    └── 📁 fixtures/         # Test data
```

## Key Components

### Core Package (`hypertrader/`)
- **`bot.py`**: Main trading bot with multi-strategy support
- **`strategies/`**: All trading strategies (technical, ML, arbitrage)
- **`binance_bots/`**: Grid trading, DCA, funding arbitrage bots
- **`risk/`**: Position sizing, drawdown control, risk management
- **`execution/`**: Order placement, CCXT integration, rate limiting
- **`data/`**: Market data fetching, WebSocket feeds, storage

### Scripts (`scripts/`)
- **Data fetching utilities**
- **System maintenance scripts**
- **Standalone tools**

### Examples (`examples/`)
- **Complete backtesting examples**
- **Strategy demonstrations**
- **Usage tutorials**

### Data (`data/`)
- **Market data files (CSV)**
- **Backtest results (JSON)**
- **Database files (SQLite)**

## Usage Patterns

### 1. Development Workflow
```bash
# 1. Fetch data
cd scripts && python simple_data_fetch.py

# 2. Test strategies  
cd examples && python real_data_backtest_example.py

# 3. Run tests
pytest -v

# 4. Deploy bot
python -m hypertrader.bot BTC-USD --config config.yaml
```

### 2. Adding New Strategies
1. Create strategy in `hypertrader/strategies/`
2. Add to `hypertrader/strategies/__init__.py`
3. Update `bot.py` initialization
4. Add tests in `tests/`

### 3. Data Management
- Raw data: `data/raw/`
- Processed data: `data/`
- Results: `data/backtest_results/`

## Benefits of This Structure

✅ **Modular**: Clear separation of concerns
✅ **Scalable**: Easy to add new components  
✅ **Maintainable**: Logical organization
✅ **Testable**: Isolated components
✅ **Professional**: Industry-standard layout