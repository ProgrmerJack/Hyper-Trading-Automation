#!/usr/bin/env python3
"""Advanced Macro and Microeconomic Indicators for Enhanced Market Analysis."""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import json

class MacroeconomicIndicators:
    """Fetches and analyzes key macroeconomic indicators."""
    
    def __init__(self, fred_api_key: Optional[str] = None):
        self.fred_api_key = fred_api_key
        self.base_url = "https://api.stlouisfed.org/fred/series/observations"
        self.cache = {}
        self.cache_duration = 3600  # 1 hour cache
        
    async def get_economic_signals(self) -> Dict[str, float]:
        """Get comprehensive economic signals affecting crypto markets."""
        
        signals = {
            'dxy_signal': 0.0,          # Dollar strength
            'inflation_signal': 0.0,     # Inflation pressure
            'yield_curve_signal': 0.0,   # Interest rate environment
            'liquidity_signal': 0.0,     # Market liquidity
            'risk_sentiment': 0.0,       # Risk-on/risk-off
            'macro_score': 0.0           # Composite macro score
        }
        
        if not self.fred_api_key:
            logging.warning("No FRED API key provided, returning neutral signals")
            return signals
        
        try:
            # Fetch key economic indicators
            indicators = await asyncio.gather(
                self._get_dxy_data(),
                self._get_inflation_data(),
                self._get_yield_curve_data(),
                self._get_money_supply_data(),
                self._get_vix_data(),
                return_exceptions=True
            )
            
            dxy_data, inflation_data, yield_data, m2_data, vix_data = indicators
            
            # Calculate individual signals
            signals['dxy_signal'] = self._analyze_dxy_trend(dxy_data) if not isinstance(dxy_data, Exception) else 0.0
            signals['inflation_signal'] = self._analyze_inflation_impact(inflation_data) if not isinstance(inflation_data, Exception) else 0.0
            signals['yield_curve_signal'] = self._analyze_yield_curve(yield_data) if not isinstance(yield_data, Exception) else 0.0
            signals['liquidity_signal'] = self._analyze_liquidity(m2_data) if not isinstance(m2_data, Exception) else 0.0
            signals['risk_sentiment'] = self._analyze_risk_sentiment(vix_data) if not isinstance(vix_data, Exception) else 0.0
            
            # Calculate composite macro score
            weights = {
                'dxy': -0.3,      # Strong dollar typically negative for crypto
                'inflation': 0.2,  # Moderate inflation can be positive (store of value)
                'yield': -0.2,     # Higher yields compete with crypto
                'liquidity': 0.4,  # More liquidity generally positive
                'risk': 0.1        # Risk sentiment affects crypto significantly
            }
            
            signals['macro_score'] = (
                signals['dxy_signal'] * weights['dxy'] +
                signals['inflation_signal'] * weights['inflation'] +
                signals['yield_curve_signal'] * weights['yield'] +
                signals['liquidity_signal'] * weights['liquidity'] +
                signals['risk_sentiment'] * weights['risk']
            )
            
        except Exception as e:
            logging.error(f"Error fetching economic indicators: {e}")
        
        return signals
    
    async def _get_fred_data(self, series_id: str, limit: int = 100) -> Optional[pd.DataFrame]:
        """Fetch data from FRED API with caching."""
        
        cache_key = f"{series_id}_{limit}"
        now = datetime.now()
        
        # Check cache
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if (now - cached_time).total_seconds() < self.cache_duration:
                return cached_data
        
        try:
            params = {
                'series_id': series_id,
                'api_key': self.fred_api_key,
                'file_type': 'json',
                'limit': limit,
                'sort_order': 'desc'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        observations = data.get('observations', [])
                        
                        if observations:
                            df = pd.DataFrame(observations)
                            df['date'] = pd.to_datetime(df['date'])
                            df['value'] = pd.to_numeric(df['value'], errors='coerce')
                            df = df.dropna().sort_values('date')
                            
                            # Cache the result
                            self.cache[cache_key] = (now, df)
                            return df
                        
        except Exception as e:
            logging.error(f"Error fetching FRED data for {series_id}: {e}")
        
        return None
    
    async def _get_dxy_data(self) -> Optional[pd.DataFrame]:
        """Get Dollar Index (DXY) data."""
        return await self._get_fred_data('DEXUSEU', 30)  # USD/EUR exchange rate as proxy
    
    async def _get_inflation_data(self) -> Optional[pd.DataFrame]:
        """Get inflation data (CPI)."""
        return await self._get_fred_data('CPIAUCSL', 12)  # Monthly CPI data
    
    async def _get_yield_curve_data(self) -> Optional[pd.DataFrame]:
        """Get 10-year treasury yield data."""
        return await self._get_fred_data('GS10', 30)  # 10-year treasury constant maturity
    
    async def _get_money_supply_data(self) -> Optional[pd.DataFrame]:
        """Get M2 money supply data."""
        return await self._get_fred_data('M2SL', 12)  # M2 money supply
    
    async def _get_vix_data(self) -> Optional[pd.DataFrame]:
        """Get VIX data for risk sentiment."""
        return await self._get_fred_data('VIXCLS', 30)  # VIX closing values
    
    def _analyze_dxy_trend(self, dxy_data: pd.DataFrame) -> float:
        """Analyze dollar strength trend."""
        if dxy_data is None or len(dxy_data) < 5:
            return 0.0
        
        try:
            # Calculate recent trend
            recent_values = dxy_data['value'].tail(5)
            trend = (recent_values.iloc[-1] - recent_values.iloc[0]) / recent_values.iloc[0]
            
            # Normalize to -1, 1 range
            return np.tanh(trend * 20)  # Amplify small changes
            
        except Exception:
            return 0.0
    
    def _analyze_inflation_impact(self, inflation_data: pd.DataFrame) -> float:
        """Analyze inflation impact on crypto."""
        if inflation_data is None or len(inflation_data) < 3:
            return 0.0
        
        try:
            # Calculate YoY inflation rate
            current = inflation_data['value'].iloc[-1]
            year_ago = inflation_data['value'].iloc[-12] if len(inflation_data) >= 12 else inflation_data['value'].iloc[0]
            
            inflation_rate = (current - year_ago) / year_ago
            
            # Moderate inflation (2-4%) slightly positive, extreme inflation negative
            if 0.02 <= inflation_rate <= 0.04:
                signal = 0.3  # Moderate positive
            elif 0.04 < inflation_rate <= 0.08:
                signal = 0.5  # Higher positive (store of value narrative)
            elif inflation_rate > 0.08:
                signal = -0.2  # Too high, negative for all assets
            elif inflation_rate < 0:
                signal = -0.3  # Deflation negative
            else:
                signal = 0.1  # Low inflation slightly positive
            
            return signal
            
        except Exception:
            return 0.0
    
    def _analyze_yield_curve(self, yield_data: pd.DataFrame) -> float:
        """Analyze interest rate environment."""
        if yield_data is None or len(yield_data) < 5:
            return 0.0
        
        try:
            current_yield = yield_data['value'].iloc[-1]
            avg_yield = yield_data['value'].tail(20).mean()
            
            # Higher yields typically negative for crypto
            yield_change = (current_yield - avg_yield) / avg_yield
            
            return -np.tanh(yield_change * 10)  # Invert and normalize
            
        except Exception:
            return 0.0
    
    def _analyze_liquidity(self, m2_data: pd.DataFrame) -> float:
        """Analyze money supply growth."""
        if m2_data is None or len(m2_data) < 6:
            return 0.0
        
        try:
            # Calculate M2 growth rate
            current = m2_data['value'].iloc[-1]
            six_months_ago = m2_data['value'].iloc[-6]
            
            growth_rate = (current - six_months_ago) / six_months_ago
            
            # Normalize growth rate (typical range 0-20% annually)
            normalized_growth = growth_rate * 2  # Convert 6-month to annual
            
            return np.tanh(normalized_growth * 5)  # Positive correlation with liquidity
            
        except Exception:
            return 0.0
    
    def _analyze_risk_sentiment(self, vix_data: pd.DataFrame) -> float:
        """Analyze market risk sentiment."""
        if vix_data is None or len(vix_data) < 5:
            return 0.0
        
        try:
            current_vix = vix_data['value'].iloc[-1]
            avg_vix = vix_data['value'].tail(20).mean()
            
            # VIX > 30 typically indicates fear, < 15 indicates complacency
            if current_vix > 30:
                sentiment = -0.5  # High fear
            elif current_vix > 25:
                sentiment = -0.2  # Moderate fear
            elif current_vix < 15:
                sentiment = 0.3   # Complacency/greed
            else:
                sentiment = 0.1   # Normal conditions
            
            # Consider trend
            vix_trend = (current_vix - avg_vix) / avg_vix
            trend_impact = -np.tanh(vix_trend * 2) * 0.3  # Rising VIX negative
            
            return sentiment + trend_impact
            
        except Exception:
            return 0.0

class MicrostructureIndicators:
    """Analyzes market microstructure for crypto trading signals."""
    
    def __init__(self):
        self.volume_profile = {}
        self.order_flow_imbalance = 0.0
        
    def analyze_microstructure(self, df: pd.DataFrame, 
                             trades_data: Optional[List[Dict]] = None) -> Dict[str, float]:
        """Analyze market microstructure indicators."""
        
        signals = {
            'volume_profile_signal': 0.0,
            'price_impact_signal': 0.0,
            'spread_signal': 0.0,
            'momentum_signal': 0.0,
            'microstructure_score': 0.0
        }
        
        if len(df) < 20:
            return signals
        
        try:
            # Volume Profile Analysis
            signals['volume_profile_signal'] = self._analyze_volume_profile(df)
            
            # Price Impact Analysis
            signals['price_impact_signal'] = self._analyze_price_impact(df)
            
            # Bid-Ask Spread Proxy (using volatility)
            signals['spread_signal'] = self._analyze_spread_proxy(df)
            
            # Momentum Persistence
            signals['momentum_signal'] = self._analyze_momentum_persistence(df)
            
            # Composite microstructure score
            weights = [0.3, 0.25, 0.2, 0.25]
            signal_values = [
                signals['volume_profile_signal'],
                signals['price_impact_signal'], 
                signals['spread_signal'],
                signals['momentum_signal']
            ]
            
            signals['microstructure_score'] = sum(w * s for w, s in zip(weights, signal_values))
            
        except Exception as e:
            logging.error(f"Error analyzing microstructure: {e}")
        
        return signals
    
    def _analyze_volume_profile(self, df: pd.DataFrame) -> float:
        """Analyze volume profile for support/resistance levels."""
        try:
            # Calculate volume-weighted average price (VWAP)
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
            
            current_price = df['close'].iloc[-1]
            current_vwap = vwap.iloc[-1]
            
            # Calculate distance from VWAP
            vwap_distance = (current_price - current_vwap) / current_vwap
            
            # Calculate volume concentration
            price_ranges = pd.cut(typical_price.tail(100), bins=10)
            volume_by_range = df['volume'].tail(100).groupby(price_ranges).sum()
            volume_concentration = volume_by_range.std() / volume_by_range.mean()
            
            # Higher concentration suggests stronger support/resistance
            concentration_signal = np.tanh(volume_concentration - 0.5)
            
            # Combine VWAP distance and concentration
            vwap_signal = np.tanh(vwap_distance * 5)
            
            return (vwap_signal * 0.7 + concentration_signal * 0.3)
            
        except Exception:
            return 0.0
    
    def _analyze_price_impact(self, df: pd.DataFrame) -> float:
        """Analyze price impact of volume."""
        try:
            # Calculate price impact as correlation between volume and price change
            returns = df['close'].pct_change().tail(50)
            volume_norm = (df['volume'] / df['volume'].rolling(20).mean()).tail(50)
            
            # Remove outliers
            returns_clean = returns[np.abs(returns) < returns.quantile(0.95)]
            volume_clean = volume_norm[returns_clean.index]
            
            if len(returns_clean) < 10:
                return 0.0
            
            correlation = returns_clean.corr(volume_clean)
            
            # High correlation suggests inefficient price discovery
            if pd.isna(correlation):
                return 0.0
            
            # Normalize correlation to trading signal
            return np.tanh(correlation * 3)
            
        except Exception:
            return 0.0
    
    def _analyze_spread_proxy(self, df: pd.DataFrame) -> float:
        """Use high-low spread as proxy for bid-ask spread."""
        try:
            # Calculate relative spread
            spread = (df['high'] - df['low']) / df['close']
            avg_spread = spread.rolling(20).mean()
            current_spread = spread.iloc[-1]
            avg_current_spread = avg_spread.iloc[-1]
            
            # Lower spread typically better for trading
            spread_signal = -np.tanh((current_spread - avg_current_spread) * 100)
            
            return spread_signal
            
        except Exception:
            return 0.0
    
    def _analyze_momentum_persistence(self, df: pd.DataFrame) -> float:
        """Analyze momentum persistence in price movements."""
        try:
            returns = df['close'].pct_change().tail(20)
            
            # Calculate momentum persistence (autocorrelation)
            persistence = returns.autocorr(lag=1)
            
            if pd.isna(persistence):
                return 0.0
            
            # Positive persistence suggests trending, negative suggests mean reversion
            return persistence
            
        except Exception:
            return 0.0

class EconomicSignalIntegrator:
    """Integrates macro and microeconomic signals into trading decisions."""
    
    def __init__(self, fred_api_key: Optional[str] = None):
        self.macro_analyzer = MacroeconomicIndicators(fred_api_key)
        self.micro_analyzer = MicrostructureIndicators()
        
    async def get_comprehensive_economic_signals(self, df: pd.DataFrame) -> Dict[str, float]:
        """Get comprehensive economic analysis for trading decisions."""
        
        try:
            # Get macro and micro signals
            macro_signals, micro_signals = await asyncio.gather(
                self.macro_analyzer.get_economic_signals(),
                asyncio.create_task(asyncio.to_thread(self.micro_analyzer.analyze_microstructure, df))
            )
            
            # Combine all signals
            combined_signals = {**macro_signals, **micro_signals}
            
            # Calculate overall economic sentiment
            economic_factors = [
                macro_signals.get('macro_score', 0.0) * 0.6,  # Macro has higher weight
                micro_signals.get('microstructure_score', 0.0) * 0.4
            ]
            
            combined_signals['economic_sentiment'] = sum(economic_factors)
            combined_signals['signal_confidence'] = min(1.0, abs(combined_signals['economic_sentiment']) + 0.1)
            
            return combined_signals
            
        except Exception as e:
            logging.error(f"Error in economic signal integration: {e}")
            return {'economic_sentiment': 0.0, 'signal_confidence': 0.0}

# Factory function for easy integration
async def get_economic_trading_signals(df: pd.DataFrame, fred_api_key: Optional[str] = None) -> Dict[str, float]:
    """Get comprehensive economic trading signals."""
    integrator = EconomicSignalIntegrator(fred_api_key)
    return await integrator.get_comprehensive_economic_signals(df)
