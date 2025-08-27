"""
Binance Historical Data Client - Replace synthetic data with real market data.
Implements proper OHLCV fetching with stylized facts preservation.
"""

import pandas as pd
import numpy as np
import requests
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import time
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class BinanceDataClient:
    """Client for fetching real Binance historical data instead of synthetic GBM."""
    
    BASE_URL = "https://api.binance.com/api/v3"
    FUTURES_URL = "https://fapi.binance.com/fapi/v1"
    
    def __init__(self, cache_dir: str = "data/binance_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        
    def get_klines(
        self, 
        symbol: str, 
        interval: str = "1m",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
        futures: bool = False
    ) -> pd.DataFrame:
        """
        Fetch OHLCV klines from Binance API.
        
        Args:
            symbol: Trading pair (e.g. 'BTCUSDT')
            interval: Kline interval ('1m', '5m', '1h', '1d')
            start_time: Start datetime
            end_time: End datetime  
            limit: Max records per request (1000 max)
            futures: Use futures API endpoint
            
        Returns:
            DataFrame with OHLCV data preserving real market stylized facts
        """
        # Check cache first
        cache_file = self._get_cache_file(symbol, interval, start_time, end_time, futures)
        if cache_file.exists():
            logger.info(f"Loading cached data: {cache_file}")
            return pd.read_parquet(cache_file)
            
        base_url = self.FUTURES_URL if futures else self.BASE_URL
        endpoint = f"{base_url}/klines"
        
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        
        if start_time:
            params["startTime"] = int(start_time.timestamp() * 1000)
        if end_time:
            params["endTime"] = int(end_time.timestamp() * 1000)
            
        try:
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                logger.warning(f"No data returned for {symbol} {interval}")
                return pd.DataFrame()
                
            # Convert to DataFrame with proper column names
            df = pd.DataFrame(data, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
                'ignore'
            ])
            
            # Convert data types and clean up
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 
                          'quote_asset_volume', 'taker_buy_base_asset_volume', 
                          'taker_buy_quote_asset_volume']
            df[numeric_cols] = df[numeric_cols].astype(float)
            df['number_of_trades'] = df['number_of_trades'].astype(int)
            
            # Convert timestamps
            df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Keep only essential OHLCV columns
            df = df[['open', 'high', 'low', 'close', 'volume', 'number_of_trades']]
            
            # Cache the result
            df.to_parquet(cache_file)
            logger.info(f"Cached {len(df)} records to {cache_file}")
            
            return df
            
        except requests.RequestException as e:
            logger.error(f"Binance API request failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Error processing Binance data: {e}")
            raise
            
    def get_historical_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str, 
        interval: str = "1m",
        futures: bool = False
    ) -> pd.DataFrame:
        """
        Fetch historical data in chunks to handle large date ranges.
        Preserves real market microstructure and stylized facts.
        """
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        all_data = []
        current_start = start_dt
        
        # Fetch in chunks to respect API limits
        while current_start < end_dt:
            current_end = min(current_start + timedelta(days=1), end_dt)
            
            chunk = self.get_klines(
                symbol=symbol,
                interval=interval,
                start_time=current_start,
                end_time=current_end,
                futures=futures
            )
            
            if not chunk.empty:
                all_data.append(chunk)
                
            current_start = current_end
            time.sleep(0.1)  # Rate limiting
            
        if not all_data:
            return pd.DataFrame()
            
        combined = pd.concat(all_data, axis=0)
        combined = combined.sort_index().drop_duplicates()
        
        logger.info(f"Fetched {len(combined)} records for {symbol} from {start_date} to {end_date}")
        return combined
        
    def _get_cache_file(
        self, 
        symbol: str, 
        interval: str,
        start_time: Optional[datetime],
        end_time: Optional[datetime],
        futures: bool
    ) -> Path:
        """Generate cache filename for data request."""
        market_type = "futures" if futures else "spot"
        start_str = start_time.strftime("%Y%m%d") if start_time else "latest"
        end_str = end_time.strftime("%Y%m%d") if end_time else "latest"
        
        filename = f"{symbol}_{interval}_{market_type}_{start_str}_{end_str}.parquet"
        return self.cache_dir / filename
        
    def get_stylized_facts_summary(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute stylized facts that real market data should exhibit:
        - Fat tails (excess kurtosis)
        - Volatility clustering  
        - Leverage effects
        - Long memory in volatility
        """
        if df.empty or len(df) < 100:
            return {}
            
        returns = df['close'].pct_change().dropna()
        abs_returns = returns.abs()
        
        # Fat tails (kurtosis should be > 3 for real markets)
        kurtosis = returns.kurtosis()
        
        # Volatility clustering (autocorr of squared returns)
        vol_clustering = abs_returns.autocorr(lag=1)
        
        # Leverage effect (corr between returns and future volatility)
        if len(returns) > 20:
            future_vol = abs_returns.shift(-1)
            leverage_effect = returns.corr(future_vol)
        else:
            leverage_effect = 0.0
            
        return {
            "excess_kurtosis": kurtosis - 3.0,
            "volatility_clustering": vol_clustering,
            "leverage_effect": leverage_effect,
            "mean_return": returns.mean(),
            "volatility": returns.std(),
            "max_drawdown": self._calculate_max_drawdown(df['close'])
        }
        
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown from price series."""
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max
        return abs(drawdowns.min())


def validate_market_data(df: pd.DataFrame, min_kurtosis: float = 1.0) -> bool:
    """
    Validate that data exhibits real market stylized facts.
    Reject synthetic-looking data that lacks fat tails, vol clustering.
    """
    if df.empty or len(df) < 100:
        return False
        
    client = BinanceDataClient()
    facts = client.get_stylized_facts_summary(df)
    
    # Check for realistic market characteristics
    checks = [
        facts.get("excess_kurtosis", 0) > min_kurtosis,  # Fat tails
        facts.get("volatility_clustering", 0) > 0.05,    # Vol clustering
        0.001 < facts.get("volatility", 0) < 0.5,        # Reasonable vol
    ]
    
    passed = sum(checks)
    logger.info(f"Market data validation: {passed}/{len(checks)} checks passed")
    logger.info(f"Stylized facts: {facts}")
    
    return passed >= len(checks) - 1  # Allow one check to fail
