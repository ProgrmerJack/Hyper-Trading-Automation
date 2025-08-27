"""
Unified Meta-Controller with Gating Logic
Implements industry-standard meta-labeling and regime gating for trading decisions.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from .regime_forecaster import EnhancedRegimeForecaster
from .sentiment_service import SentimentAggregator
from ..data.binance_client import BinanceDataClient

logger = logging.getLogger(__name__)


@dataclass
class TradingSignal:
    """Structured trading signal output."""
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # [0, 1]
    position_size: float  # Position size as fraction of capital
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    signal_components: Optional[Dict[str, float]] = None
    gates_passed: Optional[Dict[str, bool]] = None
    meta_label: Optional[bool] = None


class MetaController:
    """
    Unified meta-controller implementing:
    1. Meta-labeling to learn when to act on primary signals
    2. Multi-factor gating (regime, sentiment, risk, microstructure)
    3. Dynamic position sizing with Kelly criterion
    4. Real-time signal aggregation and filtering
    """
    
    def __init__(
        self,
        regime_forecaster: Optional[EnhancedRegimeForecaster] = None,
        sentiment_aggregator: Optional[SentimentAggregator] = None,
        meta_model_type: str = "random_forest",
        confidence_threshold: float = 0.65,
        regime_gate_threshold: float = 0.5,
        sentiment_gate_threshold: float = -0.2,
        risk_gate_params: Optional[Dict[str, float]] = None
    ):
        # Core components
        self.regime_forecaster = regime_forecaster or EnhancedRegimeForecaster()
        self.sentiment_aggregator = sentiment_aggregator
        
        # Meta-labeling model
        self.meta_model = self._initialize_meta_model(meta_model_type)
        self.meta_model_fitted = False
        
        # Gating parameters
        self.confidence_threshold = confidence_threshold
        self.regime_gate_threshold = regime_gate_threshold
        self.sentiment_gate_threshold = sentiment_gate_threshold
        self.risk_gate_params = risk_gate_params or {
            'max_var_threshold': 0.08,
            'max_volatility_threshold': 0.08,
            'max_drawdown_threshold': 0.08
        }
        
        # Signal weighting (industry standard meta-score)
        self.signal_weights = {
            'technical_weight': 0.25,
            'ml_weight': 0.25,
            'microstructure_weight': 0.20,
            'sentiment_weight': 0.15,
            'macro_weight': 0.10,
            'regime_weight': 0.05
        }
        
        # Historical performance tracking
        self.signal_history = []
        self.performance_metrics = {}
        
    def _initialize_meta_model(self, model_type: str):
        """Initialize meta-labeling model."""
        if model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42
            )
        elif model_type == "logistic_regression":
            return LogisticRegression(
                penalty='l2',
                C=1.0,
                max_iter=1000,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported meta model type: {model_type}")
    
    def fit_meta_model(
        self, 
        primary_signals: pd.DataFrame,
        outcomes: pd.Series,
        features: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Train meta-labeling model to learn when to act on primary signals.
        
        Args:
            primary_signals: Primary model predictions/probabilities
            outcomes: Actual trading outcomes (1 if profitable, 0 if not)
            features: Additional features for meta-model
            
        Returns:
            Meta-model performance metrics
        """
        if len(primary_signals) != len(outcomes):
            raise ValueError("Primary signals and outcomes must have same length")
            
        # Prepare meta-features
        meta_features = []
        
        # Add primary signal strength and confidence
        if 'confidence' in primary_signals.columns:
            meta_features.append(primary_signals['confidence'].values)
        if 'signal_strength' in primary_signals.columns:
            meta_features.append(primary_signals['signal_strength'].values)
            
        # Add regime features if available
        if hasattr(self, 'regime_scores_history'):
            regime_scores = np.array(self.regime_scores_history[-len(primary_signals):])
            if len(regime_scores) == len(primary_signals):
                meta_features.append(regime_scores)
                
        # Add sentiment features if available
        if hasattr(self, 'sentiment_scores_history'):
            sentiment_scores = np.array(self.sentiment_scores_history[-len(primary_signals):])
            if len(sentiment_scores) == len(primary_signals):
                meta_features.append(sentiment_scores)
                
        # Add custom features
        if features is not None:
            for col in features.columns:
                meta_features.append(features[col].values)
                
        if not meta_features:
            logger.warning("No features available for meta-model training")
            return {}
            
        # Combine features
        X = np.column_stack(meta_features)
        y = outcomes.values
        
        # Remove NaN values
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X_clean = X[mask]
        y_clean = y[mask]
        
        if len(X_clean) < 50:
            logger.warning("Insufficient data for meta-model training")
            return {}
            
        try:
            # Fit meta-model
            self.meta_model.fit(X_clean, y_clean)
            self.meta_model_fitted = True
            
            # Evaluate performance
            y_pred = self.meta_model.predict(X_clean)
            accuracy = accuracy_score(y_clean, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_clean, y_pred, average='binary'
            )
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'n_samples': len(X_clean)
            }
            
            logger.info(f"Meta-model trained: {metrics}")
            self.performance_metrics = metrics
            
            return metrics
            
        except Exception as e:
            logger.error(f"Meta-model training failed: {e}")
            return {}
    
    def predict_meta_label(self, signal_features: Dict[str, float]) -> Tuple[bool, float]:
        """
        Predict whether to act on primary signal using meta-model.
        
        Args:
            signal_features: Dictionary of signal features
            
        Returns:
            (should_act, confidence) tuple
        """
        if not self.meta_model_fitted:
            # Fallback to simple confidence threshold
            confidence = signal_features.get('confidence', 0.0)
            return confidence > self.confidence_threshold, confidence
            
        try:
            # Prepare feature vector (same order as training)
            feature_vector = []
            for key in ['confidence', 'signal_strength', 'regime_score', 'sentiment_score']:
                feature_vector.append(signal_features.get(key, 0.0))
                
            X = np.array(feature_vector).reshape(1, -1)
            
            # Predict probability of successful trade
            if hasattr(self.meta_model, 'predict_proba'):
                prob = self.meta_model.predict_proba(X)[0, 1]
                should_act = prob > 0.5
                return should_act, prob
            else:
                prediction = self.meta_model.predict(X)[0]
                confidence = signal_features.get('confidence', 0.5)
                return bool(prediction), confidence
                
        except Exception as e:
            logger.warning(f"Meta-model prediction failed: {e}")
            confidence = signal_features.get('confidence', 0.0)
            return confidence > self.confidence_threshold, confidence
    
    def apply_regime_gate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Apply regime-based gating to filter trades.
        
        Args:
            data: Recent market data
            
        Returns:
            Regime gate decision and metrics
        """
        try:
            regime_score_data = self.regime_forecaster.generate_regime_score(data)
            
            # Extract key regime metrics
            regime_score = regime_score_data.get('regime_score', 0.0)
            vol_regime = regime_score_data.get('volatility_regime', 'medium_vol')
            direction = regime_score_data.get('direction_forecast', 'neutral')
            
            # Gate decision logic
            regime_favorable = regime_score > self.regime_gate_threshold
            vol_acceptable = vol_regime in ['low_vol', 'medium_vol']
            directional_signal = direction != 'neutral'
            
            gate_passed = regime_favorable and vol_acceptable
            
            return {
                'gate_passed': gate_passed,
                'regime_score': regime_score,
                'volatility_regime': vol_regime,
                'direction_forecast': direction,
                'regime_favorable': regime_favorable,
                'vol_acceptable': vol_acceptable,
                'directional_signal': directional_signal
            }
            
        except Exception as e:
            logger.warning(f"Regime gate failed: {e}")
            return {
                'gate_passed': True,  # Fail open
                'regime_score': 0.5,
                'error': str(e)
            }
    
    async def apply_sentiment_gate(self, symbol: str) -> Dict[str, Any]:
        """
        Apply sentiment-based gating.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Sentiment gate decision and metrics
        """
        if self.sentiment_aggregator is None:
            return {'gate_passed': True, 'sentiment_score': 0.0}
            
        try:
            sentiment_signals = await self.sentiment_aggregator.get_sentiment_signals(symbol)
            
            sentiment_score = sentiment_signals.get('sentiment_ewma', 0.0)
            sentiment_regime = sentiment_signals.get('sentiment_regime', 'neutral')
            
            # Gate logic: Don't trade in extremely bearish sentiment
            gate_passed = sentiment_score > self.sentiment_gate_threshold
            
            return {
                'gate_passed': gate_passed,
                'sentiment_score': sentiment_score,
                'sentiment_regime': sentiment_regime,
                'bullish_signal': sentiment_signals.get('bullish_signal', 0.0),
                'bearish_signal': sentiment_signals.get('bearish_signal', 0.0),
                'catalyst_strength': sentiment_signals.get('catalyst_strength', 0.0)
            }
            
        except Exception as e:
            logger.warning(f"Sentiment gate failed: {e}")
            return {'gate_passed': True, 'sentiment_score': 0.0, 'error': str(e)}
    
    def apply_risk_gate(self, current_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Apply risk-based gating using VaR, volatility, and drawdown checks.
        
        Args:
            current_metrics: Current risk metrics (VaR, volatility, drawdown)
            
        Returns:
            Risk gate decision and metrics
        """
        gates = {}
        
        # VaR gate
        current_var = current_metrics.get('var', 0.0)
        var_gate = current_var < self.risk_gate_params['max_var_threshold']
        gates['var_gate'] = var_gate
        
        # Volatility gate
        current_vol = current_metrics.get('volatility', 0.0)
        vol_gate = current_vol < self.risk_gate_params['max_volatility_threshold']
        gates['vol_gate'] = vol_gate
        
        # Drawdown gate
        current_dd = current_metrics.get('drawdown', 0.0)
        dd_gate = current_dd < self.risk_gate_params['max_drawdown_threshold']
        gates['drawdown_gate'] = dd_gate
        
        # Overall risk gate
        overall_gate = var_gate and vol_gate and dd_gate
        gates['gate_passed'] = overall_gate
        
        return {
            **gates,
            'current_var': current_var,
            'current_vol': current_vol,
            'current_drawdown': current_dd
        }
    
    def calculate_position_size(
        self, 
        signal: Dict[str, float],
        account_balance: float,
        risk_per_trade: float = 0.02
    ) -> float:
        """
        Calculate position size using enhanced Kelly criterion with confidence scaling.
        
        Args:
            signal: Trading signal with confidence and expected return
            account_balance: Current account balance
            risk_per_trade: Maximum risk per trade as fraction of balance
            
        Returns:
            Position size as fraction of account balance
        """
        confidence = signal.get('confidence', 0.5)
        expected_return = signal.get('expected_return', 0.01)
        volatility = signal.get('volatility', 0.02)
        
        # Enhanced Kelly fraction
        if volatility > 0:
            kelly_fraction = (expected_return * confidence) / (volatility ** 2)
        else:
            kelly_fraction = 0.01
            
        # Apply safety constraints
        max_kelly = 0.25  # Never risk more than 25% on single trade
        kelly_fraction = min(kelly_fraction, max_kelly)
        
        # Scale by confidence
        confidence_adjusted = kelly_fraction * confidence
        
        # Apply maximum risk constraint
        final_size = min(confidence_adjusted, risk_per_trade)
        
        # Ensure minimum viable position
        min_size = 0.001
        final_size = max(final_size, min_size)
        
        return final_size
    
    async def generate_trading_signal(
        self,
        primary_signals: Dict[str, float],
        market_data: pd.DataFrame,
        symbol: str,
        account_balance: float,
        risk_metrics: Dict[str, float]
    ) -> TradingSignal:
        """
        Generate unified trading signal with meta-labeling and gating.
        
        Args:
            primary_signals: Primary model signals (technical, ML, etc.)
            market_data: Recent market data
            symbol: Trading symbol
            account_balance: Current account balance  
            risk_metrics: Current risk metrics
            
        Returns:
            Structured trading signal with gating results
        """
        # Step 1: Calculate meta-score
        meta_score = self._calculate_meta_score(primary_signals)
        
        # Step 2: Apply regime gate
        regime_gate = self.apply_regime_gate(market_data)
        
        # Step 3: Apply sentiment gate
        sentiment_gate = await self.apply_sentiment_gate(symbol)
        
        # Step 4: Apply risk gate
        risk_gate = self.apply_risk_gate(risk_metrics)
        
        # Step 5: Meta-labeling decision
        signal_features = {
            'confidence': primary_signals.get('confidence', 0.0),
            'signal_strength': meta_score,
            'regime_score': regime_gate.get('regime_score', 0.0),
            'sentiment_score': sentiment_gate.get('sentiment_score', 0.0)
        }
        
        should_act, meta_confidence = self.predict_meta_label(signal_features)
        
        # Step 6: Overall gating decision
        all_gates_passed = (
            regime_gate.get('gate_passed', False) and
            sentiment_gate.get('gate_passed', False) and
            risk_gate.get('gate_passed', False) and
            should_act
        )
        
        # Step 7: Generate final signal
        if all_gates_passed and meta_score > self.confidence_threshold:
            # Determine action based on signal direction
            if primary_signals.get('direction', 0) > 0:
                action = 'BUY'
            elif primary_signals.get('direction', 0) < 0:
                action = 'SELL'
            else:
                action = 'HOLD'
                
            # Calculate position size
            signal_for_sizing = {
                'confidence': meta_confidence,
                'expected_return': primary_signals.get('expected_return', 0.01),
                'volatility': regime_gate.get('volatility_forecast', 0.02)
            }
            position_size = self.calculate_position_size(signal_for_sizing, account_balance)
            
        else:
            action = 'HOLD'
            position_size = 0.0
            
        # Create structured signal
        signal = TradingSignal(
            action=action,
            confidence=meta_confidence,
            position_size=position_size,
            signal_components=primary_signals,
            gates_passed={
                'regime_gate': regime_gate.get('gate_passed', False),
                'sentiment_gate': sentiment_gate.get('gate_passed', False),
                'risk_gate': risk_gate.get('gate_passed', False),
                'meta_label': should_act,
                'overall': all_gates_passed
            },
            meta_label=should_act
        )
        
        # Log signal for performance tracking
        self._log_signal(signal, regime_gate, sentiment_gate, risk_gate)
        
        return signal
    
    def _calculate_meta_score(self, primary_signals: Dict[str, float]) -> float:
        """Calculate industry-standard meta-score from signal components."""
        score = 0.0
        
        # Technical component
        technical_score = primary_signals.get('technical_score', 0.0)
        score += self.signal_weights['technical_weight'] * technical_score
        
        # ML component
        ml_score = primary_signals.get('ml_score', 0.0)
        score += self.signal_weights['ml_weight'] * ml_score
        
        # Microstructure component
        microstructure_score = primary_signals.get('microstructure_score', 0.0)
        score += self.signal_weights['microstructure_weight'] * microstructure_score
        
        # Sentiment component
        sentiment_score = primary_signals.get('sentiment_score', 0.0)
        score += self.signal_weights['sentiment_weight'] * sentiment_score
        
        # Macro component
        macro_score = primary_signals.get('macro_score', 0.0)
        score += self.signal_weights['macro_weight'] * macro_score
        
        # Regime component
        regime_score = primary_signals.get('regime_score', 0.0)
        score += self.signal_weights['regime_weight'] * regime_score
        
        return np.clip(score, 0.0, 1.0)
    
    def _log_signal(
        self, 
        signal: TradingSignal,
        regime_gate: Dict[str, Any],
        sentiment_gate: Dict[str, Any], 
        risk_gate: Dict[str, Any]
    ) -> None:
        """Log signal for performance analysis."""
        log_entry = {
            'timestamp': pd.Timestamp.now(),
            'action': signal.action,
            'confidence': signal.confidence,
            'position_size': signal.position_size,
            'gates_passed': signal.gates_passed,
            'regime_score': regime_gate.get('regime_score', 0.0),
            'sentiment_score': sentiment_gate.get('sentiment_score', 0.0),
            'risk_gate_passed': risk_gate.get('gate_passed', False)
        }
        
        self.signal_history.append(log_entry)
        
        # Keep only last 1000 signals in memory
        if len(self.signal_history) > 1000:
            self.signal_history = self.signal_history[-1000:]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of meta-controller decisions."""
        if not self.signal_history:
            return {}
            
        df = pd.DataFrame(self.signal_history)
        
        summary = {
            'total_signals': len(df),
            'buy_signals': len(df[df['action'] == 'BUY']),
            'sell_signals': len(df[df['action'] == 'SELL']),
            'hold_signals': len(df[df['action'] == 'HOLD']),
            'avg_confidence': df['confidence'].mean(),
            'gate_pass_rate': df['gates_passed'].apply(lambda x: x.get('overall', False)).mean(),
            'regime_gate_pass_rate': df['gates_passed'].apply(lambda x: x.get('regime_gate', False)).mean(),
            'sentiment_gate_pass_rate': df['gates_passed'].apply(lambda x: x.get('sentiment_gate', False)).mean(),
            'risk_gate_pass_rate': df['gates_passed'].apply(lambda x: x.get('risk_gate', False)).mean(),
            'meta_model_metrics': self.performance_metrics
        }
        
        return summary
