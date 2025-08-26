"""Integration module for specialized Binance bot strategies."""

from typing import List, Dict, Any, Optional
import asyncio
import logging
from datetime import datetime

from ..binance_bots.spot_grid import SpotGrid
from ..binance_bots.futures_grid import FuturesGrid
from ..binance_bots.funding_arb import FundingArbBot
from ..binance_bots.spot_dca import SpotDCA
from ..binance_bots.rebalancing import RebalancingBot
from .indicator_signals import Signal

logger = logging.getLogger(__name__)


class BinanceBotManager:
    """Manager for specialized Binance trading bots."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize bot manager with configuration."""
        self.config = config
        self.active_bots = {}
        self.bot_performance = {}
        self.initialize_bots()
    
    def initialize_bots(self):
        """Initialize all configured bots."""
        try:
            # Initialize Spot Grid Bot
            if self.config.get("spot_grid", {}).get("enabled", False):
                grid_config = self.config["spot_grid"]
                self.active_bots["spot_grid"] = SpotGrid(
                    symbol=grid_config.get("symbol", "BTCUSDT"),
                    lower=grid_config.get("lower", 30000),
                    upper=grid_config.get("upper", 50000),
                    grids=grid_config.get("grids", 10),
                    base_qty=grid_config.get("base_qty", 0.001)
                )
                logger.info("Initialized Spot Grid Bot")
            
            # Initialize Futures Grid Bot
            if self.config.get("futures_grid", {}).get("enabled", False):
                futures_config = self.config["futures_grid"]
                self.active_bots["futures_grid"] = FuturesGrid(
                    symbol=futures_config.get("symbol", "BTCUSDT"),
                    lower=futures_config.get("lower", 30000),
                    upper=futures_config.get("upper", 50000),
                    grids=futures_config.get("grids", 8),
                    base_qty=futures_config.get("base_qty", 0.002)
                )
                logger.info("Initialized Futures Grid Bot")
            
            # Initialize Funding Arbitrage Bot
            if self.config.get("funding_arb", {}).get("enabled", False):
                arb_config = self.config["funding_arb"]
                self.active_bots["funding_arb"] = FundingArbBot(
                    symbol=arb_config.get("symbol", "BTCUSDT"),
                    base_qty=arb_config.get("base_qty", 0.001),
                    expected_apr_min=arb_config.get("expected_apr_min", 0.15)
                )
                logger.info("Initialized Funding Arbitrage Bot")
            
            # Initialize DCA Bot
            if self.config.get("spot_dca", {}).get("enabled", False):
                dca_config = self.config["spot_dca"]
                self.active_bots["spot_dca"] = SpotDCA(
                    symbol=dca_config.get("symbol", "BTCUSDT"),
                    base_qty=dca_config.get("base_qty", 0.001),
                    interval_minutes=dca_config.get("interval_minutes", 60)
                )
                logger.info("Initialized DCA Bot")
            
            # Initialize Rebalancing Bot
            if self.config.get("rebalancing", {}).get("enabled", False):
                rebal_config = self.config["rebalancing"]
                self.active_bots["rebalancing"] = RebalancingBot(
                    symbols=rebal_config.get("symbols", ["BTCUSDT", "ETHUSDT"]),
                    target_weights=rebal_config.get("target_weights", [0.6, 0.4]),
                    rebalance_threshold=rebal_config.get("threshold", 0.05)
                )
                logger.info("Initialized Rebalancing Bot")
                
        except Exception as e:
            logger.error(f"Failed to initialize bots: {e}")
    
    def get_bot_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        """Get signals from all active specialized bots."""
        signals = []
        
        try:
            current_price = market_data.get("price", 0)
            symbol = market_data.get("symbol", "BTCUSDT")
            
            # Grid Bot Signals - check if price is near grid levels
            if "spot_grid" in self.active_bots:
                grid_bot = self.active_bots["spot_grid"]
                grid_signal = self._evaluate_grid_signal(grid_bot, current_price)
                if grid_signal:
                    signals.append(grid_signal)
            
            if "futures_grid" in self.active_bots:
                futures_grid = self.active_bots["futures_grid"]
                futures_signal = self._evaluate_grid_signal(futures_grid, current_price, "futures")
                if futures_signal:
                    signals.append(futures_signal)
            
            # Funding Arbitrage Signals
            if "funding_arb" in self.active_bots:
                funding_signal = self._evaluate_funding_signal(
                    market_data.get("funding_rate", 0),
                    current_price
                )
                if funding_signal:
                    signals.append(funding_signal)
            
            # DCA Signals
            if "spot_dca" in self.active_bots:
                dca_signal = self._evaluate_dca_signal(market_data)
                if dca_signal:
                    signals.append(dca_signal)
            
        except Exception as e:
            logger.error(f"Error generating bot signals: {e}")
        
        return signals
    
    def _evaluate_grid_signal(self, grid_bot, current_price: float, bot_type: str = "spot") -> Optional[Signal]:
        """Evaluate grid trading signal."""
        try:
            # Find closest grid level
            distances = [abs(level - current_price) for level in grid_bot.levels]
            min_distance_idx = distances.index(min(distances))
            closest_level = grid_bot.levels[min_distance_idx]
            side = grid_bot.sides[min_distance_idx]
            
            # Signal strength based on distance to level
            distance_ratio = abs(current_price - closest_level) / current_price
            
            if distance_ratio < 0.002:  # Within 0.2% of grid level
                confidence = 0.8 - (distance_ratio * 100)  # Higher confidence when closer
                action = side if side in ['BUY', 'SELL'] else 'HOLD'
                return Signal(action, confidence=max(0.5, confidence))
                
        except Exception as e:
            logger.error(f"Grid signal evaluation error: {e}")
        
        return None
    
    def _evaluate_funding_signal(self, funding_rate: float, current_price: float) -> Optional[Signal]:
        """Evaluate funding arbitrage signal."""
        try:
            if "funding_arb" not in self.active_bots:
                return None
            
            arb_bot = self.active_bots["funding_arb"]
            estimated_apr = arb_bot.estimate_apr(funding_rate)
            
            if estimated_apr >= arb_bot.expected_apr_min:
                if funding_rate > 0.0005:  # Positive funding rate
                    return Signal('SELL', confidence=0.75)  # Short perp, long spot
                elif funding_rate < -0.0005:  # Negative funding rate
                    return Signal('BUY', confidence=0.75)  # Long perp, short spot
                    
        except Exception as e:
            logger.error(f"Funding signal evaluation error: {e}")
        
        return None
    
    def _evaluate_dca_signal(self, market_data: Dict[str, Any]) -> Optional[Signal]:
        """Evaluate DCA signal."""
        try:
            # DCA typically buys on dips
            price_change = market_data.get("price_change_24h", 0)
            
            if price_change < -0.03:  # Price down 3% or more
                return Signal('BUY', confidence=0.6)
            elif price_change < -0.05:  # Price down 5% or more
                return Signal('BUY', confidence=0.8)
                
        except Exception as e:
            logger.error(f"DCA signal evaluation error: {e}")
        
        return None
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for all bots."""
        metrics = {}
        
        for bot_name, performance in self.bot_performance.items():
            metrics[bot_name] = {
                "total_trades": performance.get("trades", 0),
                "win_rate": performance.get("win_rate", 0.0),
                "profit_loss": performance.get("pnl", 0.0),
                "last_activity": performance.get("last_activity", "Never")
            }
        
        return metrics
    
    def update_performance(self, bot_name: str, trade_result: Dict[str, Any]):
        """Update performance tracking for a bot."""
        if bot_name not in self.bot_performance:
            self.bot_performance[bot_name] = {
                "trades": 0,
                "wins": 0,
                "pnl": 0.0,
                "last_activity": datetime.now().isoformat()
            }
        
        perf = self.bot_performance[bot_name]
        perf["trades"] += 1
        perf["pnl"] += trade_result.get("pnl", 0.0)
        perf["last_activity"] = datetime.now().isoformat()
        
        if trade_result.get("pnl", 0.0) > 0:
            perf["wins"] += 1
        
        perf["win_rate"] = perf["wins"] / perf["trades"] if perf["trades"] > 0 else 0.0


class BinanceStrategyIntegrator:
    """Integrates specialized Binance bots with the main trading system."""
    
    def __init__(self, bot_manager: BinanceBotManager):
        """Initialize with bot manager."""
        self.bot_manager = bot_manager
        self.signal_weight = 0.15  # Weight for specialized bot signals
    
    def generate_signal(self, market_data: Dict[str, Any]) -> Signal:
        """Generate combined signal from specialized bots."""
        try:
            bot_signals = self.bot_manager.get_bot_signals(market_data)
            
            if not bot_signals:
                return Signal('HOLD', confidence=0.5)
            
            # Aggregate bot signals
            buy_signals = [s for s in bot_signals if s.action == 'BUY']
            sell_signals = [s for s in bot_signals if s.action == 'SELL']
            
            if len(buy_signals) > len(sell_signals):
                avg_confidence = sum(s.confidence for s in buy_signals) / len(buy_signals)
                return Signal('BUY', confidence=avg_confidence * self.signal_weight + 0.5)
            elif len(sell_signals) > len(buy_signals):
                avg_confidence = sum(s.confidence for s in sell_signals) / len(sell_signals)
                return Signal('SELL', confidence=avg_confidence * self.signal_weight + 0.5)
            else:
                return Signal('HOLD', confidence=0.5)
                
        except Exception as e:
            logger.error(f"Binance strategy integration error: {e}")
            return Signal('HOLD', confidence=0.3)


def create_binance_bot_config() -> Dict[str, Any]:
    """Create default configuration for Binance bots."""
    return {
        "spot_grid": {
            "enabled": True,
            "symbol": "BTCUSDT",
            "lower": 30000,
            "upper": 50000,
            "grids": 10,
            "base_qty": 0.001
        },
        "futures_grid": {
            "enabled": False,  # Disabled by default for safety
            "symbol": "BTCUSDT",
            "lower": 30000,
            "upper": 50000,
            "grids": 8,
            "base_qty": 0.002
        },
        "funding_arb": {
            "enabled": True,
            "symbol": "BTCUSDT",
            "base_qty": 0.001,
            "expected_apr_min": 0.15
        },
        "spot_dca": {
            "enabled": True,
            "symbol": "BTCUSDT",
            "base_qty": 0.001,
            "interval_minutes": 60
        },
        "rebalancing": {
            "enabled": False,  # Disabled by default
            "symbols": ["BTCUSDT", "ETHUSDT"],
            "target_weights": [0.6, 0.4],
            "threshold": 0.05
        }
    }
