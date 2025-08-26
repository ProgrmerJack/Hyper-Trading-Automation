"""Machine learning modules for HyperTrader.

This package provides ML-powered features including:
- Sentiment analysis and catalyst classification
- Time-series regime forecasting
- Meta scoring and signal aggregation
- Advanced backtesting with purged cross-validation
"""

from .sentiment_catalyst import compute_sentiment_and_catalyst
from .regime_forecaster import RegimeForecaster
from .meta_score import compute_meta_score, gate_entry
from .backtesting import purged_kfold_cv, walk_forward_backtest

__all__ = [
    "compute_sentiment_and_catalyst",
    "RegimeForecaster", 
    "compute_meta_score",
    "gate_entry",
    "purged_kfold_cv",
    "walk_forward_backtest",
]
