#!/usr/bin/env python3
"""
Optuna Hyperparameter Optimization for Trading Models
"""

import optuna
from hypertrader.ml.enhanced_ensemble import EnhancedMLEnsemble
from hypertrader.data.binance_client import BinanceDataClient
from hypertrader.ml.labeling import TripleBarrierLabeler

# Initialize data client
binance = BinanceDataClient()

def objective(trial):
    """Objective function for Optuna optimization"""
    
    # Get historical data
    df = binance.get_ohlcv(
        symbol='BTC-USDT', 
        timeframe='1d',
        limit=1000
    )
    
    # Create labels using triple barrier method
    labeler = TripleBarrierLabeler(
        upper_barrier=0.05,
        lower_barrier=0.03,
        vertical_barrier=5
    )
    y = labeler.fit_transform(df['close'])
    
    # Define hyperparameters to optimize
    params = {
        'use_lightgbm': True,
        'use_xgboost': True,
        'use_random_forest': trial.suggest_categorical('use_rf', [True, False]),
        'use_transformer': trial.suggest_categorical('use_transformer', [True, False]),
        'ensemble_method': trial.suggest_categorical('ensemble_method', ['weighted_average', 'voting']),
        'cv_folds': trial.suggest_int('cv_folds', 3, 10)
    }
    
    # Initialize and fit model
    model = EnhancedMLEnsemble(**params)
    cv_results = model.fit(df, y)
    
    # Return optimization target (maximize F1 score)
    return cv_results['transformer']['f1'] if 'transformer' in cv_results else cv_results['lightgbm']['f1']

def main():
    """Run hyperparameter optimization"""
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
if __name__ == "__main__":
    main()
