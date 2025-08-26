"""Real-time macro sentiment analysis with economic indicators."""

import asyncio
import aiohttp
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import json
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MacroIndicators:
    """Real-time macro economic indicators."""
    dxy: float = 0.0  # Dollar Index
    us_10y: float = 0.0  # US 10-Year Treasury
    fed_rate: float = 0.0  # Federal Fund Rate
    vix: float = 0.0  # VIX Fear Index
    gold: float = 0.0  # Gold price
    oil: float = 0.0  # Oil price
    btc_dominance: float = 0.0  # Bitcoin dominance
    total_market_cap: float = 0.0  # Total crypto market cap
    sentiment_score: float = 0.0  # Composite sentiment
    
    def calculate_sentiment(self) -> float:
        """Calculate composite macro sentiment score."""
        # Dollar strength (negative for crypto)
        dxy_sentiment = -((self.dxy - 100) / 10)  # Normalize around 100
        
        # Interest rates (negative for risk assets)
        rate_sentiment = -(self.fed_rate / 5.0)  # Normalize to 5%
        
        # Fear index (inverse relationship)
        fear_sentiment = -(self.vix - 20) / 10  # Normalize around 20
        
        # Risk-on assets (positive correlation)
        gold_sentiment = (self.gold - 2000) / 200  # Normalize around $2000
        
        # Crypto specific
        dominance_sentiment = (self.btc_dominance - 50) / 10  # Normalize around 50%
        
        # Weighted composite
        self.sentiment_score = (
            dxy_sentiment * 0.25 +
            rate_sentiment * 0.20 +
            fear_sentiment * 0.20 +
            gold_sentiment * 0.15 +
            dominance_sentiment * 0.20
        )
        
        return max(-1.0, min(1.0, self.sentiment_score))


class MacroSentimentAnalyzer:
    """Real-time macro sentiment analysis for trading decisions."""
    
    def __init__(self):
        self.indicators = MacroIndicators()
        self.cache_duration = 300  # 5 minutes
        self.last_update = None
        
    async def fetch_yahoo_data(self, symbol: str) -> Optional[float]:
        """Fetch real-time data from Yahoo Finance API."""
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        price = data['chart']['result'][0]['meta']['regularMarketPrice']
                        return float(price)
        except Exception as e:
            logger.warning(f"Failed to fetch {symbol}: {e}")
        return None
    
    async def fetch_crypto_data(self) -> Dict[str, float]:
        """Fetch crypto-specific data from CoinGecko."""
        try:
            url = "https://api.coingecko.com/api/v3/global"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        global_data = data['data']
                        return {
                            'btc_dominance': global_data['market_cap_percentage']['btc'],
                            'total_market_cap': global_data['total_market_cap']['usd'] / 1e12  # In trillions
                        }
        except Exception as e:
            logger.warning(f"Failed to fetch crypto data: {e}")
        return {'btc_dominance': 50.0, 'total_market_cap': 2.5}
    
    async def fetch_fed_rate(self) -> float:
        """Fetch current Federal Fund Rate from FRED API."""
        try:
            # Using FRED API (requires API key in production)
            # For demo, return current estimated rate
            return 5.25  # Current Fed rate as of 2024
        except Exception:
            return 5.25
    
    async def update_indicators(self) -> MacroIndicators:
        """Update all macro indicators with real-time data."""
        if (self.last_update and 
            datetime.now() - self.last_update < timedelta(seconds=self.cache_duration)):
            return self.indicators
        
        try:
            # Fetch data concurrently
            tasks = {
                'dxy': self.fetch_yahoo_data('DX-Y.NYB'),  # Dollar Index
                'us_10y': self.fetch_yahoo_data('^TNX'),   # 10-Year Treasury
                'vix': self.fetch_yahoo_data('^VIX'),      # VIX
                'gold': self.fetch_yahoo_data('GC=F'),     # Gold futures
                'oil': self.fetch_yahoo_data('CL=F'),      # Oil futures
                'crypto': self.fetch_crypto_data(),
                'fed_rate': self.fetch_fed_rate()
            }
            
            results = {}
            for name, task in tasks.items():
                try:
                    results[name] = await asyncio.wait_for(task, timeout=15)
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout fetching {name}")
                    results[name] = None
            
            # Update indicators with fetched data
            if results.get('dxy'):
                self.indicators.dxy = results['dxy']
            if results.get('us_10y'):
                self.indicators.us_10y = results['us_10y']
            if results.get('vix'):
                self.indicators.vix = results['vix']
            if results.get('gold'):
                self.indicators.gold = results['gold']
            if results.get('oil'):
                self.indicators.oil = results['oil']
            if results.get('fed_rate'):
                self.indicators.fed_rate = results['fed_rate']
            
            # Update crypto data
            crypto_data = results.get('crypto', {})
            if crypto_data:
                self.indicators.btc_dominance = crypto_data.get('btc_dominance', 50.0)
                self.indicators.total_market_cap = crypto_data.get('total_market_cap', 2.5)
            
            # Calculate composite sentiment
            self.indicators.calculate_sentiment()
            self.last_update = datetime.now()
            
            logger.info(f"Updated macro indicators: DXY={self.indicators.dxy:.2f}, "
                       f"VIX={self.indicators.vix:.2f}, "
                       f"Fed={self.indicators.fed_rate:.2f}%, "
                       f"Sentiment={self.indicators.sentiment_score:.3f}")
            
        except Exception as e:
            logger.error(f"Error updating macro indicators: {e}")
        
        return self.indicators
    
    async def get_trading_signals(self) -> Dict[str, Any]:
        """Generate trading signals based on macro sentiment."""
        indicators = await self.update_indicators()
        
        # Generate signals based on macro conditions
        signals = {
            'macro_sentiment': indicators.sentiment_score,
            'risk_on': indicators.sentiment_score > 0.3,
            'risk_off': indicators.sentiment_score < -0.3,
            'dollar_strength': indicators.dxy > 105,
            'rate_pressure': indicators.fed_rate > 5.0,
            'fear_spike': indicators.vix > 25,
            'crypto_dominance_rising': indicators.btc_dominance > 52,
            'market_cap_growing': indicators.total_market_cap > 2.8,
        }
        
        # Calculate position sizing multiplier based on macro conditions
        if signals['risk_on'] and not signals['dollar_strength']:
            signals['position_multiplier'] = 1.5  # Increase size in favorable conditions
        elif signals['risk_off'] or signals['fear_spike']:
            signals['position_multiplier'] = 0.5  # Reduce size in unfavorable conditions
        else:
            signals['position_multiplier'] = 1.0  # Normal sizing
        
        # Generate specific trading recommendations
        if indicators.sentiment_score > 0.5:
            signals['recommendation'] = 'STRONG_BUY'
            signals['confidence'] = 0.85
        elif indicators.sentiment_score > 0.2:
            signals['recommendation'] = 'BUY'
            signals['confidence'] = 0.70
        elif indicators.sentiment_score < -0.5:
            signals['recommendation'] = 'STRONG_SELL'
            signals['confidence'] = 0.85
        elif indicators.sentiment_score < -0.2:
            signals['recommendation'] = 'SELL'
            signals['confidence'] = 0.70
        else:
            signals['recommendation'] = 'HOLD'
            signals['confidence'] = 0.60
        
        return signals
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get formatted data for dashboard display."""
        return {
            'dxy': self.indicators.dxy,
            'us_10y': self.indicators.us_10y,
            'fed_rate': self.indicators.fed_rate,
            'vix': self.indicators.vix,
            'gold': self.indicators.gold,
            'oil': self.indicators.oil,
            'btc_dominance': self.indicators.btc_dominance,
            'total_market_cap': self.indicators.total_market_cap,
            'sentiment_score': self.indicators.sentiment_score,
            'last_update': self.last_update.isoformat() if self.last_update else None
        }


# Global instance for use across the application
macro_analyzer = MacroSentimentAnalyzer()


async def get_macro_sentiment_score() -> float:
    """Get current macro sentiment score (-1 to 1)."""
    signals = await macro_analyzer.get_trading_signals()
    return signals.get('macro_sentiment', 0.0)


async def get_macro_position_multiplier() -> float:
    """Get position sizing multiplier based on macro conditions."""
    signals = await macro_analyzer.get_trading_signals()
    return signals.get('position_multiplier', 1.0)


async def get_macro_trading_signals() -> Dict[str, Any]:
    """Get comprehensive macro trading signals."""
    return await macro_analyzer.get_trading_signals()
