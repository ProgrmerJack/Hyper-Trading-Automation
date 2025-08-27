"""
Triple-Barrier Labeling with Meta-Labeling
Replaces naive "next-bar up/down" with proper financial ML labels.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any
import logging
from numba import jit
from scipy import stats

logger = logging.getLogger(__name__)


@jit(nopython=True)
def _find_barrier_touch(prices: np.ndarray, start_idx: int, pt_barrier: float, 
                       sl_barrier: float, max_hold: int) -> Tuple[int, int]:
    """
    Fast barrier touch detection using Numba.
    
    Returns:
        (touch_idx, barrier_type) where barrier_type: 1=PT, -1=SL, 0=timeout
    """
    for i in range(start_idx + 1, min(len(prices), start_idx + max_hold + 1)):
        if prices[i] >= pt_barrier:
            return i, 1  # Profit take hit
        elif prices[i] <= sl_barrier:
            return i, -1  # Stop loss hit
    
    return min(len(prices) - 1, start_idx + max_hold), 0  # Timeout


class TripleBarrierLabeler:
    """
    Triple-barrier method for financial ML labeling.
    Creates labels based on first barrier touched: profit-take, stop-loss, or time-out.
    """
    
    def __init__(
        self,
        pt_multiplier: float = 2.0,
        sl_multiplier: float = 1.0,
        max_hold_periods: int = 20,
        min_ret: float = 0.005,
        vol_lookback: int = 50
    ):
        self.pt_multiplier = pt_multiplier
        self.sl_multiplier = sl_multiplier  
        self.max_hold_periods = max_hold_periods
        self.min_ret = min_ret
        self.vol_lookback = vol_lookback
        
    def create_barriers(
        self, 
        df: pd.DataFrame,
        volatility: Optional[pd.Series] = None,
        side_predictions: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Create triple-barrier labels from price data.
        
        Args:
            df: OHLCV DataFrame
            volatility: Pre-computed volatility series (uses rolling std if None)
            side_predictions: Optional primary model predictions for meta-labeling
            
        Returns:
            DataFrame with labels: y (return), t_in, t_out, side, barrier_touched
        """
        if volatility is None:
            returns = df['close'].pct_change()
            volatility = returns.rolling(self.vol_lookback).std()
            
        prices = df['close'].values
        timestamps = df.index.values
        vol_array = volatility.values
        
        results = []
        
        for i in range(len(df) - self.max_hold_periods):
            if np.isnan(vol_array[i]) or vol_array[i] <= 0:
                continue
                
            entry_price = prices[i]
            vol = vol_array[i]
            
            # Dynamic barriers based on volatility
            pt_barrier = entry_price * (1 + self.pt_multiplier * vol)
            sl_barrier = entry_price * (1 - self.sl_multiplier * vol)
            
            # Find first barrier touch
            exit_idx, barrier_type = _find_barrier_touch(
                prices, i, pt_barrier, sl_barrier, self.max_hold_periods
            )
            
            exit_price = prices[exit_idx]
            ret = (exit_price - entry_price) / entry_price
            
            # Skip if return too small (noise)
            if abs(ret) < self.min_ret:
                continue
                
            # Determine side (1 for long signals, -1 for short, 0 for no signal)
            side = 0
            if side_predictions is not None and i < len(side_predictions):
                side = side_predictions.iloc[i] if not pd.isna(side_predictions.iloc[i]) else 0
            else:
                side = 1 if ret > 0 else -1  # Simple momentum assumption
                
            results.append({
                't_in': timestamps[i],
                't_out': timestamps[exit_idx],
                'y': ret * side,  # Signed return based on side
                'side': side,
                'barrier_touched': barrier_type,
                'holding_period': exit_idx - i,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'volatility': vol
            })
            
        if not results:
            logger.warning("No valid barriers created - check parameters")
            return pd.DataFrame()
            
        labels_df = pd.DataFrame(results)
        labels_df.set_index('t_in', inplace=True)
        
        logger.info(f"Created {len(labels_df)} triple-barrier labels")
        logger.info(f"Barrier distribution - PT: {(labels_df['barrier_touched'] == 1).sum()}, "
                   f"SL: {(labels_df['barrier_touched'] == -1).sum()}, "
                   f"Timeout: {(labels_df['barrier_touched'] == 0).sum()}")
        
        return labels_df
        
    def create_meta_labels(
        self, 
        primary_signals: pd.Series,
        barriers_df: pd.DataFrame,
        threshold: float = 0.55
    ) -> pd.Series:
        """
        Create meta-labels to learn when to act on primary model signals.
        
        Args:
            primary_signals: Primary model predictions/probabilities
            barriers_df: Output from create_barriers()
            threshold: Confidence threshold for primary signals
            
        Returns:
            Binary meta-labels (1 = act on signal, 0 = ignore)
        """
        aligned_signals = primary_signals.reindex(barriers_df.index, method='nearest')
        
        # Meta-label = 1 if primary signal was confident AND profitable
        confident_signals = abs(aligned_signals) > threshold
        profitable_outcomes = barriers_df['y'] > 0
        
        meta_labels = (confident_signals & profitable_outcomes).astype(int)
        
        pos_rate = meta_labels.mean()
        logger.info(f"Meta-labeling: {pos_rate:.2%} positive labels (act on signal)")
        
        return meta_labels


class FractionalDifferencer:
    """
    Fractional differentiation for stationarity without memory loss.
    Preserves long-memory predictive content while achieving stationarity.
    """
    
    def __init__(self, d: float = 0.5, threshold: float = 1e-5):
        self.d = d
        self.threshold = threshold
        
    def get_weights(self, size: int) -> np.ndarray:
        """Compute fractional differentiation weights."""
        w = [1.0]
        for k in range(1, size):
            w_k = -w[-1] * (self.d - k + 1) / k
            if abs(w_k) < self.threshold:
                break
            w.append(w_k)
        return np.array(w)
        
    def frac_diff(self, series: pd.Series) -> pd.Series:
        """Apply fractional differentiation to achieve stationarity."""
        weights = self.get_weights(len(series))
        width = len(weights)
        
        if width >= len(series):
            logger.warning("Series too short for fractional differentiation")
            return series.copy()
            
        # Apply convolution with weights
        output = []
        for i in range(width - 1, len(series)):
            output.append(
                np.dot(weights, series.iloc[i - width + 1:i + 1].values)
            )
            
        fracdiff_series = pd.Series(
            output, 
            index=series.index[width - 1:],
            name=f"fracdiff_{series.name}"
        )
        
        return fracdiff_series
        
    def find_optimal_d(
        self, 
        series: pd.Series, 
        max_d: float = 1.0, 
        step: float = 0.05
    ) -> float:
        """Find optimal d parameter for stationarity."""
        from statsmodels.tsa.stattools import adfuller
        
        best_d = 0.0
        best_pvalue = 1.0
        
        d_values = np.arange(0.05, max_d + step, step)
        
        for d in d_values:
            self.d = d
            diff_series = self.frac_diff(series)
            
            if len(diff_series.dropna()) < 50:
                continue
                
            try:
                _, pvalue, _, _, _, _ = adfuller(diff_series.dropna())
                if pvalue < best_pvalue:
                    best_pvalue = pvalue
                    best_d = d
                    
                # Stop if we achieve stationarity
                if pvalue <= 0.01:
                    break
                    
            except Exception as e:
                logger.warning(f"ADF test failed for d={d}: {e}")
                continue
                
        logger.info(f"Optimal d={best_d:.3f} (p-value={best_pvalue:.4f})")
        self.d = best_d
        return best_d


def create_ml_features(
    df: pd.DataFrame,
    frac_diff: bool = True,
    microstructure: bool = True,
    volatility_features: bool = True
) -> pd.DataFrame:
    """
    Create proper financial ML features with fractional differentiation.
    
    Args:
        df: OHLCV DataFrame
        frac_diff: Apply fractional differentiation to prices
        microstructure: Include microstructure features
        volatility_features: Include volatility-based features
        
    Returns:
        Feature DataFrame ready for ML models
    """
    features = pd.DataFrame(index=df.index)
    
    # Price-based features
    returns = df['close'].pct_change()
    
    if frac_diff:
        # Apply fractional differentiation to achieve stationarity
        differ = FractionalDifferencer()
        differ.find_optimal_d(df['close'])
        features['fracdiff_close'] = differ.frac_diff(df['close'])
        features['fracdiff_volume'] = differ.frac_diff(df['volume'])
    else:
        features['log_returns'] = np.log(df['close']).diff()
        
    # Volatility features
    if volatility_features:
        features['realized_vol'] = returns.rolling(20).std()
        features['vol_of_vol'] = features['realized_vol'].rolling(10).std()
        features['parkinson_vol'] = np.sqrt(
            252 * np.log(df['high'] / df['low']).rolling(20).mean()
        )
        
    # Microstructure features  
    if microstructure:
        features['price_impact'] = returns / (df['volume'].rolling(5).mean() + 1e-8)
        features['bid_ask_spread_proxy'] = (df['high'] - df['low']) / df['close']
        features['trade_intensity'] = df['number_of_trades'].rolling(10).mean()
        
    # Technical features (but avoid overfitting)
    features['rsi'] = _calculate_rsi(df['close'])
    features['momentum_5'] = df['close'].pct_change(5)
    features['momentum_20'] = df['close'].pct_change(20)
    
    # Clean up
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(method='ffill').fillna(0)
    
    logger.info(f"Created {len(features.columns)} ML features")
    return features


def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))
