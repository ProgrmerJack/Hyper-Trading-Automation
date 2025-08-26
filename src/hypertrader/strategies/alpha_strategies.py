"""
Advanced Alpha Generation Strategies
High-frequency and statistical arbitrage strategies for superior returns
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

@dataclass
class AlphaSignal:
    action: str
    confidence: float
    edge: float
    strategy_type: str
    metadata: Dict

class MeanReversionStrategy:
    """Statistical mean reversion strategy."""
    
    def __init__(self):
        self.lookback = 60
        self.zscore_threshold = 2.0
        self.price_history = []
        
    def calculate_zscore(self, prices: pd.Series) -> float:
        """Calculate Z-score of current price vs historical mean."""
        if len(prices) < self.lookback:
            return 0.0
        
        recent_prices = prices.tail(self.lookback)
        mean_price = recent_prices.mean()
        std_price = recent_prices.std()
        
        if std_price == 0:
            return 0.0
            
        current_price = prices.iloc[-1]
        return (current_price - mean_price) / std_price
    
    def generate_signal(self, data: pd.DataFrame) -> AlphaSignal:
        """Generate mean reversion signal."""
        if len(data) < self.lookback:
            return AlphaSignal("HOLD", 0.5, 0.0, "mean_reversion", {})
        
        zscore = self.calculate_zscore(data['close'])
        
        # Mean reversion logic
        if zscore > self.zscore_threshold:
            # Price too high, expect reversion down
            action = "SELL"
            confidence = min(0.95, 0.6 + (abs(zscore) - self.zscore_threshold) * 0.1)
            edge = abs(zscore) * 0.5
        elif zscore < -self.zscore_threshold:
            # Price too low, expect reversion up
            action = "BUY"
            confidence = min(0.95, 0.6 + (abs(zscore) - self.zscore_threshold) * 0.1)
            edge = abs(zscore) * 0.5
        else:
            action = "HOLD"
            confidence = 0.5
            edge = 0.0
        
        metadata = {
            'zscore': zscore,
            'threshold': self.zscore_threshold,
            'lookback': self.lookback
        }
        
        return AlphaSignal(action, confidence, edge, "mean_reversion", metadata)

class MomentumBreakoutStrategy:
    """Momentum breakout strategy with volume confirmation."""
    
    def __init__(self):
        self.breakout_threshold = 0.025  # 2.5% breakout
        self.volume_multiplier = 1.5
        self.momentum_window = 20
        
    def detect_breakout(self, data: pd.DataFrame) -> Tuple[bool, str, float]:
        """Detect price breakouts with volume confirmation."""
        if len(data) < self.momentum_window:
            return False, "HOLD", 0.0
        
        recent_data = data.tail(self.momentum_window)
        current_price = data['close'].iloc[-1]
        
        # Calculate resistance and support levels
        resistance = recent_data['high'].max()
        support = recent_data['low'].min()
        avg_volume = recent_data['volume'].mean()
        current_volume = data['volume'].iloc[-1]
        
        # Volume confirmation
        volume_confirmed = current_volume > (avg_volume * self.volume_multiplier)
        
        # Breakout detection
        if current_price > resistance * (1 + self.breakout_threshold) and volume_confirmed:
            breakout_strength = (current_price - resistance) / resistance
            return True, "BUY", breakout_strength
        elif current_price < support * (1 - self.breakout_threshold) and volume_confirmed:
            breakout_strength = (support - current_price) / support
            return True, "SELL", breakout_strength
        
        return False, "HOLD", 0.0
    
    def generate_signal(self, data: pd.DataFrame) -> AlphaSignal:
        """Generate momentum breakout signal."""
        is_breakout, direction, strength = self.detect_breakout(data)
        
        if is_breakout:
            confidence = min(0.9, 0.7 + strength * 2)
            edge = strength * 10  # Convert to percentage
        else:
            direction = "HOLD"
            confidence = 0.5
            edge = 0.0
        
        metadata = {
            'breakout_detected': is_breakout,
            'breakout_strength': strength,
            'volume_confirmed': is_breakout
        }
        
        return AlphaSignal(direction, confidence, edge, "momentum_breakout", metadata)

class VolatilityTradingStrategy:
    """Volatility-based trading strategy."""
    
    def __init__(self):
        self.vol_window = 20
        self.vol_threshold_high = 0.03  # 3%
        self.vol_threshold_low = 0.01   # 1%
        
    def calculate_volatility_regime(self, data: pd.DataFrame) -> Tuple[str, float]:
        """Determine volatility regime."""
        if len(data) < self.vol_window:
            return "normal", 0.02
        
        returns = data['close'].pct_change().dropna()
        current_vol = returns.tail(self.vol_window).std()
        historical_vol = returns.std()
        
        vol_ratio = current_vol / historical_vol if historical_vol > 0 else 1.0
        
        if current_vol > self.vol_threshold_high:
            return "high", current_vol
        elif current_vol < self.vol_threshold_low:
            return "low", current_vol
        else:
            return "normal", current_vol
    
    def generate_signal(self, data: pd.DataFrame) -> AlphaSignal:
        """Generate volatility-based signal."""
        vol_regime, current_vol = self.calculate_volatility_regime(data)
        
        # Volatility trading logic
        if vol_regime == "high":
            # High volatility - expect mean reversion
            recent_return = data['close'].pct_change().iloc[-1]
            if recent_return > 0.01:  # If price moved up significantly
                action = "SELL"
                edge = current_vol * 20  # Expect reversion
            elif recent_return < -0.01:  # If price moved down significantly
                action = "BUY"
                edge = current_vol * 20
            else:
                action = "HOLD"
                edge = 0.0
            confidence = min(0.85, 0.6 + current_vol * 5)
            
        elif vol_regime == "low":
            # Low volatility - expect breakout
            action = "HOLD"  # Wait for breakout signal
            confidence = 0.5
            edge = 0.0
            
        else:
            action = "HOLD"
            confidence = 0.5
            edge = 0.0
        
        metadata = {
            'volatility_regime': vol_regime,
            'current_volatility': current_vol,
            'volatility_percentile': min(100, current_vol / 0.05 * 100)
        }
        
        return AlphaSignal(action, confidence, edge, "volatility_trading", metadata)

class PairsTradingStrategy:
    """Statistical arbitrage pairs trading."""
    
    def __init__(self, pair_symbols: List[str] = None):
        self.pair_symbols = pair_symbols or ["BTC-USD", "ETH-USD"]
        self.lookback = 60
        self.correlation_threshold = 0.8
        self.cointegration_threshold = 0.05
        
    def calculate_spread(self, price1: pd.Series, price2: pd.Series) -> pd.Series:
        """Calculate normalized spread between two price series."""
        if len(price1) != len(price2) or len(price1) < 2:
            return pd.Series([0])
        
        # Simple spread (log ratio)
        spread = np.log(price1) - np.log(price2)
        return spread
    
    def test_cointegration(self, spread: pd.Series) -> bool:
        """Test for cointegration using ADF test (simplified)."""
        if len(spread) < 20:
            return False
        
        # Simplified stationarity test - check if spread mean-reverts
        spread_mean = spread.mean()
        spread_std = spread.std()
        
        if spread_std == 0:
            return False
        
        # Check if recent values are within 2 standard deviations
        recent_spread = spread.tail(10)
        outliers = sum(abs(val - spread_mean) > 2 * spread_std for val in recent_spread)
        
        return outliers < 3  # Less than 30% outliers suggests mean reversion
    
    def generate_signal(self, data1: pd.DataFrame, data2: pd.DataFrame = None) -> AlphaSignal:
        """Generate pairs trading signal."""
        # For demo, simulate second asset data
        if data2 is None:
            # Create correlated but different price series
            data2 = data1.copy()
            data2['close'] = data1['close'] * 0.8 + np.random.normal(0, 1, len(data1)) * 5
        
        if len(data1) < self.lookback or len(data2) < self.lookback:
            return AlphaSignal("HOLD", 0.5, 0.0, "pairs_trading", {})
        
        # Calculate spread
        spread = self.calculate_spread(data1['close'], data2['close'])
        
        if len(spread) < self.lookback:
            return AlphaSignal("HOLD", 0.5, 0.0, "pairs_trading", {})
        
        # Test cointegration
        is_cointegrated = self.test_cointegration(spread)
        
        if not is_cointegrated:
            return AlphaSignal("HOLD", 0.5, 0.0, "pairs_trading", 
                             {"cointegrated": False})
        
        # Calculate z-score of spread
        spread_mean = spread.tail(self.lookback).mean()
        spread_std = spread.tail(self.lookback).std()
        
        if spread_std == 0:
            return AlphaSignal("HOLD", 0.5, 0.0, "pairs_trading", {})
        
        current_zscore = (spread.iloc[-1] - spread_mean) / spread_std
        
        # Generate signal based on spread z-score
        if current_zscore > 2:
            # Spread too high, short asset 1, long asset 2
            action = "SELL"  # Sell overvalued asset
            confidence = min(0.9, 0.6 + abs(current_zscore - 2) * 0.1)
            edge = abs(current_zscore) * 0.3
        elif current_zscore < -2:
            # Spread too low, long asset 1, short asset 2
            action = "BUY"   # Buy undervalued asset
            confidence = min(0.9, 0.6 + abs(current_zscore + 2) * 0.1)
            edge = abs(current_zscore) * 0.3
        else:
            action = "HOLD"
            confidence = 0.5
            edge = 0.0
        
        metadata = {
            'spread_zscore': current_zscore,
            'cointegrated': is_cointegrated,
            'spread_value': spread.iloc[-1]
        }
        
        return AlphaSignal(action, confidence, edge, "pairs_trading", metadata)

class OnChainAnalysisStrategy:
    """On-chain metrics-based strategy."""
    
    def __init__(self):
        self.metrics_history = []
        self.whale_threshold = 1000000  # $1M transactions
        
    def analyze_onchain_metrics(self, mock_data: Dict = None) -> Dict:
        """Analyze on-chain metrics (mocked for demo)."""
        if mock_data:
            return mock_data
        
        # Mock on-chain data
        import random
        
        metrics = {
            'active_addresses': random.randint(800000, 1200000),
            'transaction_volume': random.uniform(1e9, 5e9),
            'whale_transactions': random.randint(50, 200),
            'exchange_inflows': random.uniform(1e8, 1e9),
            'exchange_outflows': random.uniform(1e8, 1e9),
            'hodl_ratio': random.uniform(0.6, 0.8),
            'nvt_ratio': random.uniform(20, 80)
        }
        
        return metrics
    
    def generate_signal(self, market_data: pd.DataFrame, onchain_data: Dict = None) -> AlphaSignal:
        """Generate on-chain based signal."""
        metrics = self.analyze_onchain_metrics(onchain_data)
        self.metrics_history.append(metrics)
        
        score = 0
        confidence_factors = []
        
        # Analyze exchange flows
        net_flow = metrics['exchange_inflows'] - metrics['exchange_outflows']
        if net_flow < -1e8:  # Net outflow (bullish)
            score += 0.3
            confidence_factors.append(0.8)
        elif net_flow > 1e8:  # Net inflow (bearish)
            score -= 0.3
            confidence_factors.append(0.8)
        
        # Whale activity
        if metrics['whale_transactions'] > 150:
            # High whale activity can be volatile
            confidence_factors.append(0.6)
        
        # HODL behavior
        if metrics['hodl_ratio'] > 0.75:
            score += 0.2  # Strong holding (bullish)
            confidence_factors.append(0.7)
        elif metrics['hodl_ratio'] < 0.65:
            score -= 0.2  # Weak holding (bearish)
            confidence_factors.append(0.7)
        
        # NVT ratio (Network Value to Transaction)
        if metrics['nvt_ratio'] < 30:
            score += 0.2  # Undervalued (bullish)
            confidence_factors.append(0.7)
        elif metrics['nvt_ratio'] > 60:
            score -= 0.2  # Overvalued (bearish)
            confidence_factors.append(0.7)
        
        # Generate action
        if score > 0.3:
            action = "BUY"
            edge = min(5.0, score * 10)
        elif score < -0.3:
            action = "SELL"
            edge = min(5.0, abs(score) * 10)
        else:
            action = "HOLD"
            edge = 0.0
        
        confidence = np.mean(confidence_factors) if confidence_factors else 0.6
        
        return AlphaSignal(action, confidence, edge, "onchain_analysis", metrics)

# Factory function for creating alpha strategies
def create_alpha_strategy(strategy_type: str):
    """Create alpha strategy instance."""
    strategies = {
        'mean_reversion': MeanReversionStrategy,
        'momentum_breakout': MomentumBreakoutStrategy,
        'volatility_trading': VolatilityTradingStrategy,
        'pairs_trading': PairsTradingStrategy,
        'onchain_analysis': OnChainAnalysisStrategy
    }
    
    if strategy_type in strategies:
        return strategies[strategy_type]()
    else:
        return MeanReversionStrategy()  # Default
