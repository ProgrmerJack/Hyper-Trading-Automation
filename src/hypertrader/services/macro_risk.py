import pandas as pd
import numpy as np
from typing import Dict

class MacroRiskService:
    """
    Computes macro risk indicators from market data
    """
    def __init__(self):
        self.indicators = {}
        
    def update_indicators(self, data: Dict[str, float]) -> None:
        """Update macro indicators with latest data"""
        # DXY, VIX, yields, etc.
        self.indicators = data
        
    def compute_risk_flag(self) -> bool:
        """Compute risk-on/risk-off flag"""
        # Simple implementation - expand with more sophisticated logic
        vix = self.indicators.get('VIX', 20)
        dxy = self.indicators.get('DXY', 100)
        
        # Risk-on when VIX < 20 and DXY < 100
        return vix < 20 and dxy < 100
