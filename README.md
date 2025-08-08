# Hyper-Trading Automation

This project implements a simplified version of the trading automation system described in the accompanying design document.
It is a research prototype intended for educational purposes rather than a
production‑grade or high‑frequency trading engine.
See [docs/production_readiness.md](docs/production_readiness.md) for known gaps and steps required before live trading.

## Features

- Fetch OHLCV price data from cryptocurrency exchanges using [CCXT](https://github.com/ccxt/ccxt) and stream candles via its built-in WebSocket support.
- Compute technical indicators such as moving averages, EMA, SuperTrend, RSI, MACD, Bollinger Bands, VWAP, On-Balance Volume,
  ADX, Stochastic oscillator, Rate of Change, TWAP, CCI, Keltner Channels, exchange net flow, volatility clustering index,
  Fibonacci retracements, WaveTrend oscillator, multi-timeframe RSI, volume profile point of control, Ichimoku Cloud,
  Parabolic SAR and a lightweight AI momentum signal.
- Basic sentiment analysis from NewsAPI headlines using VADER.
- Macro indicators from the FRED API (DXY, interest rates, liquidity).
- Enhanced strategy uses EMA crossovers, SuperTrend direction, RSI and Bollinger Band confirmation with optional sentiment filter for more reliable signals.
- Optional machine learning model using logistic regression to predict price direction. Includes a helper to cross-validate accuracy.
- Experimental transformer-based model and reinforcement-learning utilities for dynamic leverage selection.
- Automatic position sizing based on account balance and risk percentage.
- Integrated risk controls including dynamic leverage, drawdown throttling,
  trailing stops and volatility-scaled exits with persistent equity tracking.
- Advanced risk utilities providing AI-driven historical VaR, optional
  reinforcement-learning based throttling and SHAP explainability helpers.
- Volatility scanner that ranks symbols by ATR to focus on the most
  active markets.
- Basic testing suite and GitHub Actions workflow.
- On-chain gas fee analytics and order book imbalance filters for higher quality crypto signals.
- Depth-of-market heatmap ratios to flag potential iceberg orders.
- Enhanced machine learning features include MACD histogram, VWAP distance, OBV, volatility clustering, exchange net flow,
  WaveTrend, multi-timeframe RSI and distance from volume profile POC to better capture momentum, volume and on-chain pressure.
- Resilient data fetching with retry and optional exchange fallback.
- WebSocket data ingestion via CCXT's async support and a minimal FIX
  execution skeleton for low-latency broker connectivity.
- Vectorized backtesting helper built on [vectorbt](https://github.com/vectorbt/vectorbt).
- Backtester accepts leverage to simulate capital amplification and risk.
- Multi-strategy helpers including basic arbitrage and market-making examples.
- Prop-firm style funding simulator for evaluating strategy robustness.
- Dockerfile for containerised deployment.
- Prometheus metrics and IsolationForest anomaly detection utilities for
  monitoring and compliance.
- Configuration schemas validated with ``pydantic``.

### Macro data

Providing a FRED API key allows the bot to fetch the US Dollar Index, federal
funds rate and M2 money stock. These values are combined into a macroeconomic
score that influences trading signals.

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

### Configuration

Runtime options such as default trading symbols and risk parameters are stored
in `config.yaml`. A `config.sample.yaml` is provided as a template – copy it to
`config.yaml` and adjust values for your environment. API keys are **not**
stored in the YAML file and should instead be supplied via environment
variables:

```bash
export FRED_API_KEY=your_fred_key
export NEWS_API_KEY=your_news_key
export ETHERSCAN_API_KEY=your_etherscan_key
```

You may also pass a custom config path to the bot using `--config`.

2. Run tests:

```bash
pytest -v
```
3. Example backtest with performance metrics:

```bash
python tests/backtest.py
```

When not operating live, the bot writes decisions to `signal.json` for offline inspection.

### Running the autonomous bot

The `hypertrader.bot` module fetches data from supported exchanges via CCXT, optionally gathers news sentiment, and when run in live mode places orders directly via CCXT. When not live it writes a trading signal with calculated position size to `signal.json`. When multiple symbols are provided the bot will automatically trade the one with the highest recent volatility:

```bash
python -m hypertrader.bot BTC-USD ETH-USD --account_balance 10000 --risk_percent 5 \
     --fred_api_key YOUR_FRED_KEY \
     --model_path trained_model.pkl

# alternatively load options from config.yaml
python -m hypertrader.bot --config config.yaml

 ```

Alternatively set the `FRED_API_KEY` environment variable so the bot can
retrieve macroeconomic series without specifying the flag each run.

### Logging

The bot emits structured JSON logs with latency and other metrics. Redirect
stdout to a file or log processor to monitor live trading performance.

### Machine learning strategy

You can optionally train a simple logistic regression model on historical data.
The helper functions in `hypertrader.strategies.ml_strategy` make this easy:

```python
from hypertrader.strategies.ml_strategy import train_model, ml_signal
from hypertrader.data.fetch_data import fetch_ohlcv

data = fetch_ohlcv("binance", "BTC/USDT", timeframe="1m")
model = train_model(data)
sig = ml_signal(model, data)
print(sig)
model.to_pickle("trained_model.pkl")

# Evaluate accuracy using cross validation
from hypertrader.strategies.ml_strategy import cross_validate_model
print("CV", cross_validate_model(data))
The bot can load this model to confirm indicator-based signals for added
confidence.

Run the bot with the `--model_path` option to enable this behaviour:

```bash
python -m hypertrader.bot BTC-USD --model_path trained_model.pkl
```
