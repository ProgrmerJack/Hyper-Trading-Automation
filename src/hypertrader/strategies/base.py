"""
Base strategy class for the hypertrader framework.

This module defines the base class that all trading strategies should inherit from.
It provides a common interface and shared functionality for strategy implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""
    
    id: str
    name: str
    description: str = ""
    config: Optional[Dict[str, Any]] = None
    
    @abstractmethod
    def update(self, *args, **kwargs) -> Any:
        """
        Update the strategy with new market data.
        
        This method should be implemented by all concrete strategy classes.
        
        Returns
        -------
        Any
            Strategy-specific output (signals, orders, etc.)
        """
        pass
    
    def initialize(self) -> None:
        """
        Initialize the strategy.
        
        This method can be overridden by subclasses to perform any
        initialization logic required when the strategy is first created.
        """
        pass
    
    def reset(self) -> None:
        """
        Reset the strategy state.
        
        This method can be overridden by subclasses to reset any
        internal state variables.
        """
        pass
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get the current parameters of the strategy.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary of parameter names and values
        """
        return {}
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """
        Set the parameters of the strategy.
        
        Parameters
        ----------
        params : Dict[str, Any]
            Dictionary of parameter names and values
        """
        pass
