import pytest
from src.hypertrader.bot import TradingBot
from unittest.mock import MagicMock

class TestTradingBot:
    @pytest.fixture
    def mock_bot(self):
        """Create a mock TradingBot instance"""
        connector = MagicMock()
        strategy = MagicMock()
        cost_model = MagicMock()
        return TradingBot(connector, strategy, 'BTC-USD', cost_model=cost_model)
        
    def test_cost_model_integration(self, mock_bot):
        """Test that cost model is properly integrated"""
        assert hasattr(mock_bot, 'cost_model')
        
    def test_balance_verification(self, mock_bot):
        """Test balance verification during trade execution"""
        mock_bot.cost_model.verify_balance.return_value = False
        mock_bot.on_new_tick(50000, [])
        assert mock_bot.connector.create_order.call_count == 0
