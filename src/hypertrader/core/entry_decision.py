import time
from typing import Dict, List, Optional
from hypertrader.services.macro_risk import MacroRiskService
from hypertrader.services.indicators import IndicatorService
from hypertrader.services.ml_service import MLService
from hypertrader.core.strategy_registry import StrategyRegistry


class EntryDecisionSystem:
    def __init__(self, config: Dict):
        self.config = config
        self.cooldowns: Dict[str, float] = {}  # symbol: expiration timestamp

    def calculate_meta_score(self, symbol: str, 
                            strategy_scores: Dict[str, float],
                            ml_signals: Dict[str, float],
                            macro_flags: Dict[str, bool],
                            sentiment_score: float) -> float:
        """Calculate the meta-score for a trading symbol.
        
        Args:
            symbol: Trading symbol
            strategy_scores: Dictionary of strategy scores
            ml_signals: Dictionary of ML model signals
            macro_flags: Dictionary of macroeconomic flags
            sentiment_score: Overall sentiment score
            
        Returns:
            Meta-score between 0 and 1
        """
        # Extract weights from config
        weights = self.config['ml']['meta_score']
        
        # Calculate component scores
        technical_avg = sum(strategy_scores.values()) / len(strategy_scores) if strategy_scores else 0
        ml_avg = sum(ml_signals.values()) / len(ml_signals) if ml_signals else 0
        
        # Macro score is 1 if risk-on, 0 if risk-off
        macro_score = 1.0 if macro_flags.get('risk_on', False) else 0
        
        # Apply weights
        score = (
            weights['technical_weight'] * technical_avg +
            weights['ml_weight'] * ml_avg +
            weights['sentiment_weight'] * sentiment_score +
            weights['macro_weight'] * macro_score
        )
        
        # Apply confidence boost if enabled
        if self.config['signals'].get('confidence_boost', False):
            score *= 1.1
            score = min(score, 1.0)  # Cap at 1.0
            
        return score

    def validate_quorum(self, meta_score: float, strategy_scores: Dict[str, float]) -> bool:
        """Validate if the trade meets quorum requirements.
        
        Args:
            meta_score: Computed meta-score
            strategy_scores: Individual strategy scores
            
        Returns:
            True if quorum is met, False otherwise
        """
        # Check meta-score against threshold
        signals_config = self.config['signals']
        if meta_score < signals_config['weak_threshold']:
            return False
            
        # Check strategy consensus
        passing_strategies = [
            score >= signals_config['medium_threshold'] 
            for score in strategy_scores.values()
        ]
        consensus = sum(passing_strategies) / len(passing_strategies) if passing_strategies else 0
        
        return consensus >= signals_config['consensus_required']

    def check_cooldown(self, symbol: str) -> bool:
        """Check if a symbol is in cooldown period.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            True if cooldown is active, False otherwise
        """
        current_time = time.time()
        cooldown_end = self.cooldowns.get(symbol, 0)
        return current_time < cooldown_end

    def set_cooldown(self, symbol: str, duration: int):
        """Set cooldown for a symbol.
        
        Args:
            symbol: Trading symbol
            duration: Cooldown duration in seconds
        """
        self.cooldowns[symbol] = time.time() + duration

    def check_hard_gates(self, 
                        symbol: str, 
                        indicators: Dict[str, float],
                        macro_flags: Dict[str, bool]) -> bool:
        """Check hard gates for a trading symbol.
        
        Args:
            symbol: Trading symbol
            indicators: Dictionary of technical indicators
            macro_flags: Dictionary of macroeconomic flags
            
        Returns:
            True if all hard gates pass, False otherwise
        """
        # Check macro risk flag
        if not macro_flags.get('risk_on', False):
            return False
            
        # Check volatility threshold
        if indicators.get('volatility', 0) > self.config['risk']['max_volatility']:
            return False
            
        # Check drawdown
        if indicators.get('drawdown', 0) > self.config['risk']['max_drawdown']:
            return False
            
        return True

    def make_entry_decision(self, 
                           symbol: str, 
                           strategy_scores: Dict[str, float],
                           ml_signals: Dict[str, float],
                           sentiment_score: float,
                           indicators: Dict[str, float],
                           macro_flags: Dict[str, bool]) -> bool:
        """Make final entry decision for a symbol.
        
        Args:
            symbol: Trading symbol
            strategy_scores: Strategy confidence scores
            ml_signals: ML model signals
            sentiment_score: Overall sentiment score
            indicators: Technical indicators
            macro_flags: Macroeconomic flags
            
        Returns:
            True if entry is approved, False otherwise
        """
        # Check cooldown first
        if self.check_cooldown(symbol):
            return False
            
        # Check hard gates
        if not self.check_hard_gates(symbol, indicators, macro_flags):
            return False
            
        # Calculate meta-score
        meta_score = self.calculate_meta_score(
            symbol, 
            strategy_scores,
            ml_signals,
            macro_flags,
            sentiment_score
        )
        
        # Validate quorum
        if not self.validate_quorum(meta_score, strategy_scores):
            return False
            
        # Apply entry balance factor
        if self.config['signals'].get('entry_balance_factor', 1.0) < 1.0:
            # Implement balanced entry timing logic
            # (e.g., only enter during certain market phases)
            # Placeholder for actual implementation
            pass
            
        return True
