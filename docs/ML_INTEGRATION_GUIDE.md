# HuggingFace ML Integration Guide

This guide documents the machine learning enhancements added to HyperTrader, including sentiment analysis, regime forecasting, and advanced backtesting capabilities.

## Overview

The ML integration introduces four key modules under `hypertrader/ml/`:

1. **sentiment_catalyst.py** - Asynchronous sentiment analysis using FinBERT and Twitter RoBERTa
2. **regime_forecaster.py** - Time-series forecasting with HuggingFace transformers
3. **meta_score.py** - Signal aggregation and entry gating
4. **backtesting.py** - Purged cross-validation and walk-forward testing

## Quick Start

### Installation

Install the additional ML dependencies:

```bash
pip install optimum[onnxruntime]>=1.16.0 datasets>=2.16.0 accelerate>=0.26.0 tokenizers>=0.15.0
```

### Basic Usage

```python
from hypertrader.ml import (
    compute_sentiment_and_catalyst,
    RegimeForecaster,
    compute_meta_score,
    gate_entry
)

# Sentiment analysis
headlines = ["Company reports strong earnings", "Market volatility increases"]
tweets = []
factors = await compute_sentiment_and_catalyst(headlines, tweets)
sentiment_score = factors["fin_sent_logit"]

# Regime forecasting
forecaster = RegimeForecaster(model_name="google/timesfm-1.0-200m", use_onnx=True)
regime_value = forecaster.forecast(price_series)
regime_label = forecaster.classify_regime(regime_value)

# Meta scoring
meta_score = compute_meta_score(
    micro_score=0.1,
    tech_score=0.2,
    sentiment_score=sentiment_score,
    regime_score=regime_value
)

# Entry gating
allow_trade = gate_entry(sentiment_score, regime_label)
```

## Module Details

### Sentiment Analysis (`sentiment_catalyst.py`)

Provides asynchronous sentiment analysis using multiple HuggingFace models:

- **FinBERT**: Financial news sentiment (ProsusAI/finbert)
- **Twitter RoBERTa**: Social media sentiment
- **BART-MNLI**: Zero-shot catalyst classification

Key functions:
- `compute_sentiment_and_catalyst()` - Main entry point
- `compute_finbert_scores()` - Financial sentiment
- `compute_twitter_scores()` - Social sentiment
- `compute_catalyst_tags()` - Event classification

### Regime Forecasting (`regime_forecaster.py`)

Time-series forecasting using transformer models with optional ONNX acceleration:

```python
forecaster = RegimeForecaster(
    model_name="google/timesfm-1.0-200m",
    horizon=1,
    use_onnx=True
)

# Forecast next period
forecast_value = forecaster.forecast(price_series)
regime = forecaster.classify_regime(forecast_value)
```

### Meta Scoring (`meta_score.py`)

Combines multiple signals into unified scores:

```python
# Weighted combination of factors
meta_score = compute_meta_score(
    micro_score=book_skew,
    tech_score=technical_signals,
    sentiment_score=sentiment,
    regime_score=regime_forecast,
    weights={"micro": 0.3, "tech": 0.2, "sentiment": 0.3, "regime": 0.2}
)

# Entry gating
allow_entry = gate_entry(sentiment_score, regime_label, threshold=0.1)
```

### Advanced Backtesting (`backtesting.py`)

Prevents overfitting with proper cross-validation:

```python
from hypertrader.ml.backtesting import purged_kfold_cv, walk_forward_backtest

# Purged K-Fold (prevents leakage)
splitter = purged_kfold_cv(n_splits=5, embargo=10)
for train_idx, test_idx in splitter(features):
    model.fit(features.iloc[train_idx], labels.iloc[train_idx])
    # Evaluate on test set

# Walk-forward backtesting
avg_sharpe, avg_drawdown = walk_forward_backtest(
    model, features, labels, initial_window=200, step=50
)
```

## Integration with Bot

The ML modules are integrated into `bot.py` with the following enhancements:

### 1. Sentiment Analysis Integration

```python
# Replace basic sentiment with ML-powered analysis
factor_dict = await compute_sentiment_and_catalyst(headlines, tweets)
sentiment = factor_dict["fin_sent_logit"]
catalyst_macro = factor_dict.get("macro-CPI", 0.0)
```

### 2. Regime Forecasting

```python
# Global regime model to avoid reloading
regime_model = get_regime_model()
regime_value = regime_model.forecast(data["close"])
regime_label = regime_model.classify_regime(regime_value)
```

### 3. Meta Scoring

```python
# Combine all factors
meta = compute_meta_score(
    micro_score=book_skew,
    tech_score=tech_score,
    sentiment_score=sentiment,
    regime_score=regime_value
)
```

### 4. Entry Gating

```python
# Gate entries based on sentiment and regime
allow_entry = gate_entry(sentiment_score=sentiment, regime_label=regime_label)
if not allow_entry:
    for s in strategy_signals.values():
        s["action"] = "HOLD"
```

## Demo vs Live Mode Parity

The integration ensures consistent behavior between demo and live modes:

1. **Unified Order Logic**: Same risk checks and gating for both modes
2. **Realistic Simulation**: Demo mode uses actual market prices and meta scores
3. **Consistent Risk Management**: Same fee/slippage gates and position sizing

## Performance Optimization

### ONNX Acceleration

Enable ONNX runtime for faster inference:

```python
forecaster = RegimeForecaster(use_onnx=True)
```

### Model Caching

Models are cached globally to avoid reloading:

```python
# Cached in bot.py
_regime_model = None
def get_regime_model():
    global _regime_model
    if _regime_model is None:
        _regime_model = RegimeForecaster(...)
    return _regime_model
```

## Best Practices

### 1. Overfitting Prevention

- Use `purged_kfold_cv` for hyperparameter tuning
- Apply `walk_forward_backtest` for performance validation
- Limit feature sets to avoid curse of dimensionality

### 2. Latency Management

- Cache models at startup
- Use ONNX runtime for production
- Consider model quantization for extreme low-latency

### 3. Fallback Handling

All ML functions gracefully fallback to defaults if models fail to load:

```python
try:
    sentiment = await compute_sentiment_and_catalyst(headlines, tweets)
except Exception:
    sentiment = compute_sentiment_score(headlines)  # Fallback
```

## Examples

See `examples/ml_backtesting_example.py` for a complete demonstration of the ML backtesting capabilities.

## Troubleshooting

### Model Loading Issues

1. Check internet connection for model downloads
2. Verify HuggingFace transformers installation
3. Check available disk space for model cache

### Performance Issues

1. Enable ONNX runtime: `use_onnx=True`
2. Reduce model complexity or use smaller variants
3. Cache models globally to avoid reloading

### Memory Issues

1. Use model quantization with ONNX
2. Reduce batch sizes for sentiment analysis
3. Clear model cache periodically if needed

## Future Enhancements

1. **Twitter API Integration**: Add real-time social sentiment
2. **Custom Model Training**: Train domain-specific models
3. **Multi-Asset Forecasting**: Extend to portfolio-level predictions
4. **Real-time Model Updates**: Continuous learning capabilities
