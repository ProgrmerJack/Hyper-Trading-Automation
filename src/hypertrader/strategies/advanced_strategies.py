#!/usr/bin/env python3
"""Advanced Trading Strategies Using Sophisticated Technical Indicators."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from ..indicators.advanced_indicators import (
    stochastic_oscillator, williams_r, commodity_channel_index, parabolic_sar,
    ichimoku_cloud, volume_oscillator, chaikin_money_flow, on_balance_volume,
    aroon_indicator, average_directional_index, keltner_channels, price_channel,
    ultimate_oscillator, vortex_indicator
)

class MultiOscillatorStrategy:
    """Combines multiple oscillators for high-confidence signals."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.name = "multi_oscillator"
        self.oversold_threshold = self.config.get('oversold_threshold', 25)
        self.overbought_threshold = self.config.get('overbought_threshold', 75)
        self.consensus_required = self.config.get('consensus_required', 3)
    
    def generate_signals(self, df: pd.DataFrame) -> Dict[str, float]:
        """Generate signals using multiple oscillators with consensus voting."""
        if len(df) < 50:
            return {'signal': 0.0, 'confidence': 0.0}
        
        signals = []
        confidences = []
        
        # Stochastic Oscillator
        try:
            k_percent, d_percent = stochastic_oscillator(df['high'], df['low'], df['close'])
            stoch_signal = 0.0
            if k_percent.iloc[-1] < self.oversold_threshold and d_percent.iloc[-1] < self.oversold_threshold:
                stoch_signal = 1.0  # Buy
            elif k_percent.iloc[-1] > self.overbought_threshold and d_percent.iloc[-1] > self.overbought_threshold:
                stoch_signal = -1.0  # Sell
            
            signals.append(stoch_signal)
            confidences.append(abs(stoch_signal) * 0.8)
        except:
            pass
        
        # Williams %R
        try:
            wr = williams_r(df['high'], df['low'], df['close'])
            wr_signal = 0.0
            if wr.iloc[-1] < -80:  # Oversold
                wr_signal = 1.0
            elif wr.iloc[-1] > -20:  # Overbought
                wr_signal = -1.0
            
            signals.append(wr_signal)
            confidences.append(abs(wr_signal) * 0.7)
        except:
            pass
        
        # Commodity Channel Index
        try:
            cci = commodity_channel_index(df['high'], df['low'], df['close'])
            cci_signal = 0.0
            if cci.iloc[-1] < -100:  # Oversold
                cci_signal = 1.0
            elif cci.iloc[-1] > 100:  # Overbought
                cci_signal = -1.0
            
            signals.append(cci_signal)
            confidences.append(abs(cci_signal) * 0.75)
        except:
            pass
        
        # Ultimate Oscillator
        try:
            uo = ultimate_oscillator(df['high'], df['low'], df['close'])
            uo_signal = 0.0
            if uo.iloc[-1] < 30:  # Oversold
                uo_signal = 1.0
            elif uo.iloc[-1] > 70:  # Overbought
                uo_signal = -1.0
            
            signals.append(uo_signal)
            confidences.append(abs(uo_signal) * 0.85)
        except:
            pass
        
        if not signals:
            return {'signal': 0.0, 'confidence': 0.0}
        
        # Calculate consensus
        buy_votes = sum(1 for s in signals if s > 0)
        sell_votes = sum(1 for s in signals if s < 0)
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        final_signal = 0.0
        if buy_votes >= self.consensus_required:
            final_signal = 1.0
        elif sell_votes >= self.consensus_required:
            final_signal = -1.0
        
        return {
            'signal': final_signal,
            'confidence': avg_confidence * (max(buy_votes, sell_votes) / len(signals)),
            'component_votes': {'buy': buy_votes, 'sell': sell_votes, 'neutral': len(signals) - buy_votes - sell_votes}
        }

class IchimokuStrategy:
    """Advanced Ichimoku Cloud strategy for comprehensive trend analysis."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.name = "ichimoku_cloud"
    
    def generate_signals(self, df: pd.DataFrame) -> Dict[str, float]:
        """Generate signals using Ichimoku Cloud components."""
        if len(df) < 60:
            return {'signal': 0.0, 'confidence': 0.0}
        
        try:
            ichimoku = ichimoku_cloud(df['high'], df['low'], df['close'])
            
            current_price = df['close'].iloc[-1]
            tenkan = ichimoku['tenkan_sen'].iloc[-1]
            kijun = ichimoku['kijun_sen'].iloc[-1]
            span_a = ichimoku['senkou_span_a'].iloc[-1]
            span_b = ichimoku['senkou_span_b'].iloc[-1]
            
            # Cloud analysis
            cloud_top = max(span_a, span_b) if not pd.isna(span_a) and not pd.isna(span_b) else None
            cloud_bottom = min(span_a, span_b) if not pd.isna(span_a) and not pd.isna(span_b) else None
            
            signal = 0.0
            confidence = 0.0
            
            # Strong bullish conditions
            if (tenkan > kijun and 
                current_price > cloud_top and 
                cloud_top is not None):
                signal = 1.0
                confidence = 0.9
            
            # Strong bearish conditions
            elif (tenkan < kijun and 
                  current_price < cloud_bottom and 
                  cloud_bottom is not None):
                signal = -1.0
                confidence = 0.9
            
            # Moderate bullish (above cloud but mixed signals)
            elif current_price > cloud_top and cloud_top is not None:
                signal = 0.5
                confidence = 0.6
            
            # Moderate bearish (below cloud but mixed signals)
            elif current_price < cloud_bottom and cloud_bottom is not None:
                signal = -0.5
                confidence = 0.6
            
            return {
                'signal': signal,
                'confidence': confidence,
                'tenkan_kijun_cross': 1 if tenkan > kijun else -1,
                'price_cloud_position': 'above' if current_price > cloud_top else 'below' if current_price < cloud_bottom else 'inside'
            }
            
        except Exception as e:
            return {'signal': 0.0, 'confidence': 0.0, 'error': str(e)}

class TrendStrengthStrategy:
    """Uses ADX and other trend strength indicators for high-quality trend trades."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.name = "trend_strength"
        self.adx_threshold = self.config.get('adx_threshold', 25)
        self.strong_trend_threshold = self.config.get('strong_trend_threshold', 40)
    
    def generate_signals(self, df: pd.DataFrame) -> Dict[str, float]:
        """Generate signals based on trend strength analysis."""
        if len(df) < 30:
            return {'signal': 0.0, 'confidence': 0.0}
        
        try:
            # ADX and Directional Indicators
            adx, plus_di, minus_di = average_directional_index(df['high'], df['low'], df['close'])
            
            # Aroon Indicator
            aroon_up, aroon_down = aroon_indicator(df['high'], df['low'])
            
            # Vortex Indicator
            vi_plus, vi_minus = vortex_indicator(df['high'], df['low'], df['close'])
            
            current_adx = adx.iloc[-1]
            current_plus_di = plus_di.iloc[-1]
            current_minus_di = minus_di.iloc[-1]
            current_aroon_up = aroon_up.iloc[-1]
            current_aroon_down = aroon_down.iloc[-1]
            current_vi_plus = vi_plus.iloc[-1]
            current_vi_minus = vi_minus.iloc[-1]
            
            # Check for strong trend conditions
            if current_adx < self.adx_threshold:
                return {'signal': 0.0, 'confidence': 0.0, 'trend_strength': 'weak'}
            
            signal = 0.0
            confidence = 0.0
            trend_strength = 'moderate'
            
            # Strong uptrend conditions
            bullish_signals = 0
            if current_plus_di > current_minus_di:
                bullish_signals += 1
            if current_aroon_up > current_aroon_down and current_aroon_up > 70:
                bullish_signals += 1
            if current_vi_plus > current_vi_minus and current_vi_plus > 1.1:
                bullish_signals += 1
            
            # Strong downtrend conditions
            bearish_signals = 0
            if current_minus_di > current_plus_di:
                bearish_signals += 1
            if current_aroon_down > current_aroon_up and current_aroon_down > 70:
                bearish_signals += 1
            if current_vi_minus > current_vi_plus and current_vi_minus > 1.1:
                bearish_signals += 1
            
            # Determine signal strength
            if current_adx > self.strong_trend_threshold:
                trend_strength = 'strong'
                confidence_multiplier = 1.0
            else:
                confidence_multiplier = 0.7
            
            if bullish_signals >= 2:
                signal = 1.0
                confidence = (bullish_signals / 3) * confidence_multiplier
            elif bearish_signals >= 2:
                signal = -1.0
                confidence = (bearish_signals / 3) * confidence_multiplier
            
            return {
                'signal': signal,
                'confidence': confidence,
                'trend_strength': trend_strength,
                'adx_value': current_adx,
                'bullish_indicators': bullish_signals,
                'bearish_indicators': bearish_signals
            }
            
        except Exception as e:
            return {'signal': 0.0, 'confidence': 0.0, 'error': str(e)}

class VolumeProfileStrategy:
    """Uses volume-based indicators for institutional flow analysis."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.name = "volume_profile"
        self.cmf_threshold = self.config.get('cmf_threshold', 0.05)
        self.vo_threshold = self.config.get('vo_threshold', 5.0)
    
    def generate_signals(self, df: pd.DataFrame) -> Dict[str, float]:
        """Generate signals based on volume analysis."""
        if len(df) < 25:
            return {'signal': 0.0, 'confidence': 0.0}
        
        try:
            # Chaikin Money Flow
            cmf = chaikin_money_flow(df['high'], df['low'], df['close'], df['volume'])
            
            # Volume Oscillator
            vo = volume_oscillator(df['volume'])
            
            # On-Balance Volume
            obv = on_balance_volume(df['close'], df['volume'])
            
            current_cmf = cmf.iloc[-1]
            current_vo = vo.iloc[-1]
            
            # OBV trend analysis
            obv_sma = obv.rolling(10).mean()
            obv_trend = 1 if obv.iloc[-1] > obv_sma.iloc[-1] else -1
            
            signal = 0.0
            confidence = 0.0
            
            # Strong buying pressure
            if (current_cmf > self.cmf_threshold and 
                current_vo > self.vo_threshold and 
                obv_trend > 0):
                signal = 1.0
                confidence = 0.85
            
            # Strong selling pressure
            elif (current_cmf < -self.cmf_threshold and 
                  current_vo < -self.vo_threshold and 
                  obv_trend < 0):
                signal = -1.0
                confidence = 0.85
            
            # Moderate signals
            elif current_cmf > self.cmf_threshold or (current_vo > self.vo_threshold and obv_trend > 0):
                signal = 0.5
                confidence = 0.6
            elif current_cmf < -self.cmf_threshold or (current_vo < -self.vo_threshold and obv_trend < 0):
                signal = -0.5
                confidence = 0.6
            
            return {
                'signal': signal,
                'confidence': confidence,
                'cmf_value': current_cmf,
                'volume_oscillator': current_vo,
                'obv_trend': obv_trend
            }
            
        except Exception as e:
            return {'signal': 0.0, 'confidence': 0.0, 'error': str(e)}

class BreakoutConfirmationStrategy:
    """Uses multiple breakout indicators with volume confirmation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.name = "breakout_confirmation"
        self.lookback_period = self.config.get('lookback_period', 20)
        self.volume_multiplier = self.config.get('volume_multiplier', 1.5)
    
    def generate_signals(self, df: pd.DataFrame) -> Dict[str, float]:
        """Generate breakout signals with multiple confirmations."""
        if len(df) < self.lookback_period + 5:
            return {'signal': 0.0, 'confidence': 0.0}
        
        try:
            # Price Channels (Donchian)
            upper_channel, lower_channel = price_channel(df['high'], df['low'], self.lookback_period)
            
            # Keltner Channels
            kelt_upper, kelt_middle, kelt_lower = keltner_channels(df['high'], df['low'], df['close'])
            
            # Parabolic SAR
            sar = parabolic_sar(df['high'], df['low'], df['close'])
            
            current_price = df['close'].iloc[-1]
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].rolling(20).mean().iloc[-1]
            
            volume_confirmation = current_volume > (avg_volume * self.volume_multiplier)
            
            signal = 0.0
            confidence = 0.0
            breakout_type = None
            
            # Bullish breakout conditions
            bullish_breakouts = 0
            if current_price > upper_channel.iloc[-1]:
                bullish_breakouts += 1
            if current_price > kelt_upper.iloc[-1]:
                bullish_breakouts += 1
            if current_price > sar.iloc[-1]:
                bullish_breakouts += 1
            
            # Bearish breakout conditions
            bearish_breakouts = 0
            if current_price < lower_channel.iloc[-1]:
                bearish_breakouts += 1
            if current_price < kelt_lower.iloc[-1]:
                bearish_breakouts += 1
            if current_price < sar.iloc[-1]:
                bearish_breakouts += 1
            
            # Determine signal
            if bullish_breakouts >= 2:
                signal = 1.0
                confidence = 0.8 if volume_confirmation else 0.6
                breakout_type = 'bullish'
            elif bearish_breakouts >= 2:
                signal = -1.0
                confidence = 0.8 if volume_confirmation else 0.6
                breakout_type = 'bearish'
            elif bullish_breakouts >= 1 and volume_confirmation:
                signal = 0.5
                confidence = 0.6
                breakout_type = 'weak_bullish'
            elif bearish_breakouts >= 1 and volume_confirmation:
                signal = -0.5
                confidence = 0.6
                breakout_type = 'weak_bearish'
            
            return {
                'signal': signal,
                'confidence': confidence,
                'breakout_type': breakout_type,
                'volume_confirmation': volume_confirmation,
                'bullish_indicators': bullish_breakouts,
                'bearish_indicators': bearish_breakouts
            }
            
        except Exception as e:
            return {'signal': 0.0, 'confidence': 0.0, 'error': str(e)}

# Strategy registry for easy access
ADVANCED_STRATEGIES = {
    'multi_oscillator': MultiOscillatorStrategy,
    'ichimoku_cloud': IchimokuStrategy,
    'trend_strength': TrendStrengthStrategy,
    'volume_profile': VolumeProfileStrategy,
    'breakout_confirmation': BreakoutConfirmationStrategy
}

def get_strategy(name: str, config: Optional[Dict[str, Any]] = None):
    """Get strategy instance by name."""
    if name not in ADVANCED_STRATEGIES:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(ADVANCED_STRATEGIES.keys())}")
    
    return ADVANCED_STRATEGIES[name](config)

def get_all_strategies(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Get all advanced strategy instances."""
    return {name: cls(config) for name, cls in ADVANCED_STRATEGIES.items()}
