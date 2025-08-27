#!/usr/bin/env python3
"""
Performance Monitoring Pipeline
"""

import time
import pandas as pd
from prometheus_client import start_http_server, Gauge, Counter
from hypertrader.data.binance_client import BinanceDataClient
from hypertrader.ml.enhanced_ensemble import EnhancedMLEnsemble

class PerformanceMonitor:
    def __init__(self, port: int = 8000):
        # Initialize metrics
        self.metrics = {
            'portfolio_value': Gauge('portfolio_value', 'Current portfolio value'),
            'daily_return': Gauge('daily_return', 'Daily return percentage'),
            'model_accuracy': Gauge('model_accuracy', 'ML model accuracy'),
            'trade_count': Counter('trade_count', 'Total trades executed'),
            'win_rate': Gauge('win_rate', 'Strategy win rate'),
            'sharpe_ratio': Gauge('sharpe_ratio', 'Risk-adjusted performance'),
            'max_drawdown': Gauge('max_drawdown', 'Maximum drawdown percentage'),
            'model_latency': Gauge('model_latency', 'ML prediction latency in ms')
        }
        
        self.data_client = BinanceDataClient()
        self.model = EnhancedMLEnsemble()
        start_http_server(port)
    
    def update_trading_metrics(self, portfolio: dict):
        """Update trading performance metrics"""
        self.metrics['portfolio_value'].set(portfolio['value'])
        self.metrics['daily_return'].set(portfolio['daily_return'])
        self.metrics['trade_count'].inc(portfolio['trades_today'])
        self.metrics['win_rate'].set(portfolio['win_rate'])
        self.metrics['sharpe_ratio'].set(portfolio['sharpe'])
        self.metrics['max_drawdown'].set(portfolio['max_dd'])
    
    def update_model_metrics(self, df: pd.DataFrame):
        """Update ML model performance metrics"""
        start_time = time.time()
        
        # Get latest predictions
        predictions = self.model.predict(df)
        
        # Calculate metrics
        latency = (time.time() - start_time) * 1000
        accuracy = sum(1 for p in predictions if p.confidence > 0.7) / len(predictions)
        
        self.metrics['model_accuracy'].set(accuracy)
        self.metrics['model_latency'].set(latency)
    
    def run(self):
        """Main monitoring loop"""
        while True:
            try:
                # Get portfolio data (simulated for demo)
                portfolio = {
                    'value': 1050.42,
                    'daily_return': 1.2,
                    'trades_today': 3,
                    'win_rate': 0.65,
                    'sharpe': 1.8,
                    'max_dd': 2.3
                }
                
                # Get market data
                df = self.data_client.get_ohlcv('BTC-USDT', '1h', 100)
                
                # Update metrics
                self.update_trading_metrics(portfolio)
                self.update_model_metrics(df)
                
                time.sleep(60)  # Update every minute
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(10)

def main():
    """Start performance monitoring"""
    monitor = PerformanceMonitor()
    print("Performance monitoring started on port 8000")
    print("Metrics available at http://localhost:8000")
    monitor.run()

if __name__ == "__main__":
    main()
