import os
from typing import Dict, Any
from hypertrader.config import load_config

class TradingEnvironment:
    """
    Manages trading environment configuration including demo mode and exchange endpoints.
    """
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.demo_mode = self.config.get('demo_mode', False)
        self.binance_endpoints = self.config.get('binance_endpoints', {})
        self.state_json_path = self.config.get('state_json_path')
        self.oms_db_path = self.config.get('oms_db_path')

    def get_binance_rest_endpoint(self) -> str:
        """Returns the appropriate Binance REST endpoint based on demo mode."""
        if self.demo_mode:
            return self.binance_endpoints.get('futures_testnet') if \
                self.config.get('trading_type') == 'futures' else self.binance_endpoints.get('spot_testnet')
        else:
            # Mainnet endpoints (to be added to config)
            return self.binance_endpoints.get('futures_mainnet') if \
                self.config.get('trading_type') == 'futures' else self.binance_endpoints.get('spot_mainnet')

    def get_binance_websocket_endpoint(self) -> str:
        """Returns the appropriate Binance WebSocket endpoint based on demo mode."""
        if self.demo_mode:
            return self.binance_endpoints.get('websocket_testnet')
        else:
            return self.binance_endpoints.get('websocket_mainnet')

    def get_state_json_path(self) -> str:
        return self.state_json_path

    def get_oms_db_path(self) -> str:
        return self.oms_db_path
