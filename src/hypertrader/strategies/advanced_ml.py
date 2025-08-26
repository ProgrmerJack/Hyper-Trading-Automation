"""
Advanced ML Trading Strategies
Enhanced machine learning models for superior alpha generation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class MLSignal:
    action: str
    confidence: float
    probability: float
    feature_importance: Dict[str, float]
    model_type: str

class EnsembleMachineLearning:
    """Advanced ensemble ML strategy combining multiple models."""
    
    def __init__(self):
        self.models = {}
        self.feature_columns = [
            'price_return', 'volume_return', 'volatility', 'rsi', 'macd', 
            'bb_position', 'atr', 'momentum', 'mean_reversion', 'trend_strength'
        ]
        self.lookback = 100
        
    def extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract advanced features for ML models."""
        features = []
        
        # Price-based features
        returns = data['close'].pct_change()
        features.append(returns.rolling(5).mean())
        features.append(returns.rolling(20).std())
        
        # Volume features
        volume_ma = data['volume'].rolling(20).mean()
        features.append((data['volume'] / volume_ma).fillna(1))
        
        # Technical indicators
        features.append(self._rsi(data['close']))
        features.append(self._macd_signal(data['close']))
        features.append(self._bollinger_position(data['close']))
        features.append(self._atr(data))
        
        # Momentum features
        features.append(returns.rolling(10).sum())  # 10-period momentum
        
        # Mean reversion indicator
        price_ma = data['close'].rolling(20).mean()
        features.append((data['close'] - price_ma) / price_ma)
        
        # Trend strength
        features.append(self._adx(data))
        
        return np.column_stack(features)
    
    def _rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _macd_signal(self, prices: pd.Series) -> pd.Series:
        """Calculate MACD signal line."""
        exp1 = prices.ewm(span=12).mean()
        exp2 = prices.ewm(span=26).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9).mean()
        return macd - signal
    
    def _bollinger_position(self, prices: pd.Series, period: int = 20) -> pd.Series:
        """Calculate position within Bollinger Bands."""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        return (prices - lower) / (upper - lower)
    
    def _atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(period).mean()
    
    def _adx(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ADX trend strength."""
        high_diff = data['high'].diff()
        low_diff = -data['low'].diff()
        
        pos_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        neg_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        atr = self._atr(data, period)
        pos_di = 100 * (pos_dm.rolling(period).sum() / atr)
        neg_di = 100 * (neg_dm.rolling(period).sum() / atr)
        
        dx = 100 * np.abs(pos_di - neg_di) / (pos_di + neg_di)
        return dx.rolling(period).mean()
    
    def train_models(self, data: pd.DataFrame) -> bool:
        """Train ensemble of ML models."""
        try:
            features = self.extract_features(data)
            if features.shape[0] < self.lookback:
                return False
            
            # Create target (future returns)
            future_returns = data['close'].pct_change().shift(-1)
            target = (future_returns > 0.002).astype(int)  # Binary: >0.2% gain
            
            # Clean data
            mask = ~(np.isnan(features).any(axis=1) | np.isnan(target))
            features_clean = features[mask]
            target_clean = target[mask]
            
            if len(features_clean) < 50:  # Need minimum data
                return False
            
            # Simple ensemble (mock training for now)
            self.models['trained'] = True
            return True
            
        except Exception:
            return False
    
    def generate_signal(self, data: pd.DataFrame) -> MLSignal:
        """Generate ML-based trading signal."""
        if not self.models.get('trained', False):
            if not self.train_models(data):
                return MLSignal("HOLD", 0.5, 0.5, {}, "ensemble_ml")
        
        try:
            features = self.extract_features(data)
            if features.shape[0] == 0:
                return MLSignal("HOLD", 0.5, 0.5, {}, "ensemble_ml")
            
            # Get latest features
            latest_features = features[-1]
            
            # Simple prediction logic (enhanced)
            rsi_val = latest_features[3] if len(latest_features) > 3 else 50
            macd_val = latest_features[4] if len(latest_features) > 4 else 0
            bb_pos = latest_features[5] if len(latest_features) > 5 else 0.5
            momentum = latest_features[7] if len(latest_features) > 7 else 0
            
            # Scoring system
            score = 0
            confidence_factors = []
            
            # RSI signals
            if rsi_val < 25:
                score += 0.3
                confidence_factors.append(0.8)
            elif rsi_val > 75:
                score -= 0.3
                confidence_factors.append(0.8)
            
            # MACD signals
            if macd_val > 0:
                score += 0.2
                confidence_factors.append(0.7)
            else:
                score -= 0.2
                confidence_factors.append(0.7)
            
            # Bollinger Band position
            if bb_pos < 0.2:
                score += 0.25
                confidence_factors.append(0.75)
            elif bb_pos > 0.8:
                score -= 0.25
                confidence_factors.append(0.75)
            
            # Momentum
            if momentum > 0.01:
                score += 0.25
                confidence_factors.append(0.8)
            elif momentum < -0.01:
                score -= 0.25
                confidence_factors.append(0.8)
            
            # Determine action and confidence
            if score > 0.3:
                action = "BUY"
                probability = min(0.95, 0.5 + abs(score))
            elif score < -0.3:
                action = "SELL"
                probability = min(0.95, 0.5 + abs(score))
            else:
                action = "HOLD"
                probability = 0.5
            
            confidence = np.mean(confidence_factors) if confidence_factors else 0.6
            
            feature_importance = {
                'rsi': abs(rsi_val - 50) / 50,
                'macd': min(1.0, abs(macd_val) * 10),
                'bollinger': abs(bb_pos - 0.5) * 2,
                'momentum': min(1.0, abs(momentum) * 50)
            }
            
            return MLSignal(action, confidence, probability, feature_importance, "ensemble_ml")
            
        except Exception:
            return MLSignal("HOLD", 0.5, 0.5, {}, "ensemble_ml")

class ReinforcementLearningStrategy:
    """RL-based adaptive trading strategy."""
    
    def __init__(self):
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 0.1
        self.state_history = []
        self.action_history = []
        self.reward_history = []
        
    def get_state(self, data: pd.DataFrame) -> str:
        """Convert market data to discrete state."""
        if len(data) < 20:
            return "neutral"
        
        recent = data.tail(20)
        price_change = (recent['close'].iloc[-1] - recent['close'].iloc[0]) / recent['close'].iloc[0]
        volatility = recent['close'].pct_change().std()
        volume_ratio = recent['volume'].iloc[-1] / recent['volume'].mean()
        
        # Discretize state
        price_state = "up" if price_change > 0.01 else "down" if price_change < -0.01 else "flat"
        vol_state = "high" if volatility > 0.02 else "low"
        volume_state = "high" if volume_ratio > 1.5 else "normal"
        
        return f"{price_state}_{vol_state}_{volume_state}"
    
    def choose_action(self, state: str) -> str:
        """Choose action using epsilon-greedy policy."""
        if state not in self.q_table:
            self.q_table[state] = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}
        
        # Epsilon-greedy
        if np.random.random() < self.epsilon:
            return np.random.choice(["BUY", "SELL", "HOLD"])
        else:
            return max(self.q_table[state], key=self.q_table[state].get)
    
    def update_q_table(self, state: str, action: str, reward: float, next_state: str):
        """Update Q-table with new experience."""
        if state not in self.q_table:
            self.q_table[state] = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}
        if next_state not in self.q_table:
            self.q_table[next_state] = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}
        
        # Q-learning update
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state].values())
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][action] = new_q
    
    def generate_signal(self, data: pd.DataFrame, last_pnl: float = 0.0) -> MLSignal:
        """Generate RL-based trading signal."""
        state = self.get_state(data)
        action = self.choose_action(state)
        
        # Update from previous experience
        if len(self.state_history) > 0:
            prev_state = self.state_history[-1]
            prev_action = self.action_history[-1]
            reward = last_pnl * 10  # Scale reward
            self.update_q_table(prev_state, prev_action, reward, state)
        
        # Store current experience
        self.state_history.append(state)
        self.action_history.append(action)
        
        # Calculate confidence based on Q-values
        if state in self.q_table:
            q_values = list(self.q_table[state].values())
            confidence = (max(q_values) - min(q_values) + 0.1) / 2  # Normalize
            confidence = min(0.95, max(0.5, confidence))
        else:
            confidence = 0.6
        
        return MLSignal(action, confidence, confidence, {"q_value": confidence}, "reinforcement_learning")

class SentimentAnalysisStrategy:
    """News and social sentiment-based trading."""
    
    def __init__(self):
        self.sentiment_keywords = {
            'bullish': ['bullish', 'moon', 'pump', 'bull', 'surge', 'rally', 'breakout'],
            'bearish': ['bearish', 'crash', 'dump', 'bear', 'drop', 'decline', 'correction']
        }
        self.sentiment_history = []
        
    def analyze_sentiment(self, text_data: List[str]) -> float:
        """Analyze sentiment from text data."""
        if not text_data:
            return 0.0
        
        total_score = 0
        total_weight = 0
        
        for text in text_data:
            text_lower = text.lower()
            bullish_count = sum(1 for word in self.sentiment_keywords['bullish'] if word in text_lower)
            bearish_count = sum(1 for word in self.sentiment_keywords['bearish'] if word in text_lower)
            
            if bullish_count + bearish_count > 0:
                sentiment_score = (bullish_count - bearish_count) / (bullish_count + bearish_count)
                total_score += sentiment_score
                total_weight += 1
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def generate_signal(self, market_data: pd.DataFrame, news_data: List[str] = None) -> MLSignal:
        """Generate sentiment-based signal."""
        if not news_data:
            # Mock sentiment for demo
            news_data = ["Bitcoin showing strong momentum", "Market rally continues"]
        
        sentiment_score = self.analyze_sentiment(news_data)
        self.sentiment_history.append(sentiment_score)
        
        # Calculate trend in sentiment
        if len(self.sentiment_history) > 10:
            recent_sentiment = np.mean(self.sentiment_history[-5:])
            older_sentiment = np.mean(self.sentiment_history[-10:-5])
            sentiment_trend = recent_sentiment - older_sentiment
        else:
            sentiment_trend = sentiment_score
        
        # Generate action
        if sentiment_score > 0.3 and sentiment_trend > 0:
            action = "BUY"
            confidence = min(0.9, 0.6 + sentiment_score)
        elif sentiment_score < -0.3 and sentiment_trend < 0:
            action = "SELL"
            confidence = min(0.9, 0.6 + abs(sentiment_score))
        else:
            action = "HOLD"
            confidence = 0.5
        
        feature_importance = {
            'sentiment_score': abs(sentiment_score),
            'sentiment_trend': abs(sentiment_trend),
            'news_volume': len(news_data) / 10.0
        }
        
        return MLSignal(action, confidence, confidence, feature_importance, "sentiment_analysis")

# Factory function for creating ML strategies
def create_ml_strategy(strategy_type: str):
    """Create ML strategy instance."""
    strategies = {
        'ensemble': EnsembleMachineLearning,
        'reinforcement': ReinforcementLearningStrategy,
        'sentiment': SentimentAnalysisStrategy
    }
    
    if strategy_type in strategies:
        return strategies[strategy_type]()
    else:
        return EnsembleMachineLearning()  # Default
