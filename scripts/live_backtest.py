#!/usr/bin/env python3
"""
Live Backtesting with Binance Data
"""

import pandas as pd
from hypertrader.ml.enhanced_ensemble import EnhancedMLEnsemble
from hypertrader.data.binance_client import BinanceDataClient
from hypertrader.ml.labeling import TripleBarrierLabeler
from hypertrader.ml.cross_validation import PurgedKFold
from hypertrader.risk.manager import RiskManager

class LiveBacktester:
    def __init__(self):
        self.data_client = BinanceDataClient()
        self.risk_manager = RiskManager(
            max_daily_loss=0.1,
            max_position=0.3,
            fee_rate=0.001
        )
        
    def fetch_data(self, symbol: str, timeframe: str, limit: int = 1000):
        """Fetch OHLCV data from Binance"""
        return self.data_client.get_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            limit=limit
        )
        
    def run_backtest(self, symbol: str, initial_balance: float = 1000.0):
        """Run complete backtest with ML ensemble"""
        # Get historical data
        df = self.fetch_data(symbol, '1d', 2000)
        
        # Create labels
        labeler = TripleBarrierLabeler(
            upper_barrier=0.05,
            lower_barrier=0.03,
            vertical_barrier=5
        )
        y = labeler.fit_transform(df['close'])
        
        # Initialize ML ensemble
        model = EnhancedMLEnsemble(
            use_lightgbm=True,
            use_xgboost=True,
            use_transformer=True,
            ensemble_method='weighted_average'
        )
        
        # Purged cross-validation
        cv = PurgedKFold(n_splits=5)
        results = []
        
        for train_idx, test_idx in cv.split(df, y):
            X_train, X_test = df.iloc[train_idx], df.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Test predictions
            predictions = model.predict(X_test)
            
            # Simulate trades with risk management
            portfolio = initial_balance
            for i, (_, row) in enumerate(X_test.iterrows()):
                if i >= len(predictions):
                    continue
                    
                pred = predictions[i]
                if pred.prediction > 0.6:  # Strong buy signal
                    size = self.risk_manager.calculate_position_size(
                        portfolio_value=portfolio,
                        risk_pct=0.05,
                        entry_price=row['close'],
                        stop_loss=row['close'] * 0.97
                    )
                    portfolio += (X_test.iloc[i+1]['close'] - row['close']) * size if i+1 < len(X_test) else 0
                
            results.append(portfolio)
            
        return {
            'final_balance': sum(results)/len(results),
            'return_pct': (sum(results)/len(results) - initial_balance)/initial_balance * 100,
            'num_trades': len(df),
            'win_rate': len([x for x in results if x > initial_balance])/len(results)
        }

def main():
    """Run backtest and print results"""
    backtester = LiveBacktester()
    results = backtester.run_backtest('BTC-USDT')
    
    print("Backtest Results:")
    print(f"  Initial Balance: $1000.00")
    print(f"  Final Balance: ${results['final_balance']:.2f}")
    print(f"  Return: {results['return_pct']:.2f}%")
    print(f"  Win Rate: {results['win_rate']*100:.2f}%")
    
if __name__ == "__main__":
    main()
