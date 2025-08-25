#!/usr/bin/env python3
"""Example demonstrating ML backtesting with purged cross-validation.

This script shows how to use the new ML backtesting utilities to validate
trading strategies while avoiding overfitting through proper cross-validation
and walk-forward testing.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from hypertrader.ml.backtesting import purged_kfold_cv, walk_forward_backtest


def create_sample_data(n_samples=1000):
    """Create sample trading data for demonstration."""
    np.random.seed(42)
    
    # Generate synthetic price data
    returns = np.random.normal(0.001, 0.02, n_samples)
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Create features (technical indicators)
    features = pd.DataFrame({
        'price': prices,
        'sma_5': pd.Series(prices).rolling(5).mean(),
        'sma_20': pd.Series(prices).rolling(20).mean(),
        'rsi': pd.Series(prices).pct_change().rolling(14).apply(
            lambda x: 100 - (100 / (1 + (x[x > 0].mean() / abs(x[x < 0].mean()))))
        ),
        'volatility': pd.Series(prices).pct_change().rolling(10).std(),
    }).fillna(method='bfill')
    
    # Create binary labels (1 for up, 0 for down)
    future_returns = pd.Series(prices).pct_change().shift(-1)
    labels = (future_returns > 0).astype(int)
    
    return features.iloc[:-1], labels.iloc[:-1]


def main():
    """Demonstrate ML backtesting capabilities."""
    print("Creating sample trading data...")
    features, labels = create_sample_data()
    
    print(f"Data shape: {features.shape}")
    print(f"Features: {list(features.columns)}")
    print(f"Label distribution: {labels.value_counts().to_dict()}")
    
    # Example 1: Purged K-Fold Cross-Validation
    print("\n=== Purged K-Fold Cross-Validation ===")
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    splitter = purged_kfold_cv(n_splits=5, embargo=10)
    
    cv_scores = []
    for fold, (train_idx, test_idx) in enumerate(splitter(features)):
        print(f"Fold {fold + 1}: Train size={len(train_idx)}, Test size={len(test_idx)}")
        
        X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
        y_train, y_test = labels.iloc[train_idx], labels.iloc[test_idx]
        
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        cv_scores.append(score)
        print(f"  Accuracy: {score:.3f}")
    
    print(f"Mean CV Accuracy: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
    
    # Example 2: Walk-Forward Backtesting
    print("\n=== Walk-Forward Backtesting ===")
    
    model = LogisticRegression(random_state=42)
    avg_sharpe, avg_drawdown = walk_forward_backtest(
        model=model,
        features=features,
        labels=labels,
        initial_window=200,
        step=50
    )
    
    print(f"Average Sharpe Ratio: {avg_sharpe:.3f}")
    print(f"Average Drawdown: {avg_drawdown:.3f}")
    
    # Example 3: Feature Importance Analysis
    print("\n=== Feature Importance ===")
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(features, labels)
    
    importance_df = pd.DataFrame({
        'feature': features.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(importance_df)
    
    print("\n=== Recommendations ===")
    print("1. Use purged_kfold_cv for hyperparameter tuning to avoid leakage")
    print("2. Use walk_forward_backtest for realistic performance estimation")
    print("3. Monitor feature importance to avoid overfitting")
    print("4. Consider dimensionality reduction if using many features")
    print("5. Validate on out-of-sample data before going live")


if __name__ == "__main__":
    main()
