import pytest
from hypertrader.cost import CostModel

class TestCostModel:
    def test_estimate_cost(self):
        """Test cost estimation logic"""
        model = CostModel()
        cost = model.estimate_cost('BTC-USD', 'buy', 1.0, 50000)
        assert cost > 0
        
    def test_insufficient_balance(self):
        """Test balance verification logic"""
        model = CostModel()
        assert model.verify_balance('BTC-USD', 'buy', 1.0, 50000, {'USD': 40000}) is False
        assert model.verify_balance('BTC-USD', 'buy', 1.0, 50000, {'USD': 60000}) is True
