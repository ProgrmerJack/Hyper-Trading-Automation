from typing import Dict, List
from hypertrader.strategies.base import BaseStrategy

class StrategyRegistry:
    """
    Manages strategy instances with unique IDs
    """
    def __init__(self):
        self.strategies: Dict[str, BaseStrategy] = {}
        
    def register(self, strategy: BaseStrategy) -> None:
        """Register a strategy instance"""
        if strategy.id in self.strategies:
            raise ValueError(f"Strategy ID {strategy.id} already exists")
        self.strategies[strategy.id] = strategy
        
    def get_strategy(self, strategy_id: str) -> BaseStrategy:
        """Get strategy by ID"""
        return self.strategies.get(strategy_id)
        
    def get_all_strategies(self) -> List[BaseStrategy]:
        """Get all registered strategies"""
        return list(self.strategies.values())
        
    def load_all(self, config: dict) -> None:
        """Load all strategies from configuration"""
        # Implementation to instantiate strategies from config
        pass
