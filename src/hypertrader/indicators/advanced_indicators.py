#!/usr/bin/env python3
"""Advanced Technical Indicators for Enhanced Trading Profitability."""

import pandas as pd
import numpy as np
from typing import Tuple, Optional

def stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, 
                         k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    """
    Stochastic Oscillator - measures momentum and overbought/oversold conditions.
    
    Returns:
    - %K line: Fast stochastic
    - %D line: Slow stochastic (SMA of %K)
    """
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_period).mean()
    
    return k_percent, d_percent

def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Williams %R - momentum oscillator measuring overbought/oversold levels.
    Values range from -100 to 0.
    """
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    
    wr = -100 * ((highest_high - close) / (highest_high - lowest_low))
    return wr

def commodity_channel_index(high: pd.Series, low: pd.Series, close: pd.Series, 
                           period: int = 20) -> pd.Series:
    """
    Commodity Channel Index (CCI) - measures deviation from statistical mean.
    Values above +100 indicate overbought, below -100 indicate oversold.
    """
    typical_price = (high + low + close) / 3
    sma_tp = typical_price.rolling(window=period).mean()
    mean_deviation = typical_price.rolling(window=period).apply(
        lambda x: np.abs(x - x.mean()).mean()
    )
    
    cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
    return cci

def parabolic_sar(high: pd.Series, low: pd.Series, close: pd.Series,
                  acceleration: float = 0.02, maximum: float = 0.2) -> pd.Series:
    """
    Parabolic SAR - trend-following indicator providing stop and reverse points.
    """
    sar = pd.Series(index=close.index, dtype=float)
    trend = pd.Series(index=close.index, dtype=int)
    af = acceleration
    
    # Initialize
    sar.iloc[0] = low.iloc[0]
    trend.iloc[0] = 1  # 1 for uptrend, -1 for downtrend
    ep = high.iloc[0]  # Extreme point
    
    for i in range(1, len(close)):
        if trend.iloc[i-1] == 1:  # Uptrend
            sar.iloc[i] = sar.iloc[i-1] + af * (ep - sar.iloc[i-1])
            
            if low.iloc[i] <= sar.iloc[i]:
                # Trend reversal
                trend.iloc[i] = -1
                sar.iloc[i] = ep
                ep = low.iloc[i]
                af = acceleration
            else:
                trend.iloc[i] = 1
                if high.iloc[i] > ep:
                    ep = high.iloc[i]
                    af = min(af + acceleration, maximum)
        else:  # Downtrend
            sar.iloc[i] = sar.iloc[i-1] - af * (sar.iloc[i-1] - ep)
            
            if high.iloc[i] >= sar.iloc[i]:
                # Trend reversal
                trend.iloc[i] = 1
                sar.iloc[i] = ep
                ep = high.iloc[i]
                af = acceleration
            else:
                trend.iloc[i] = -1
                if low.iloc[i] < ep:
                    ep = low.iloc[i]
                    af = min(af + acceleration, maximum)
    
    return sar

def ichimoku_cloud(high: pd.Series, low: pd.Series, close: pd.Series) -> dict:
    """
    Ichimoku Cloud components - comprehensive trend analysis system.
    
    Returns dictionary with:
    - tenkan_sen: Conversion line (9-period)
    - kijun_sen: Base line (26-period)
    - senkou_span_a: Leading span A
    - senkou_span_b: Leading span B (52-period)
    - chikou_span: Lagging span
    """
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (high.rolling(9).max() + low.rolling(9).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (high.rolling(26).max() + low.rolling(26).min()) / 2
    
    # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    
    # Chikou Span (Lagging Span): Close shifted back 26 periods
    chikou_span = close.shift(-26)
    
    return {
        'tenkan_sen': tenkan_sen,
        'kijun_sen': kijun_sen,
        'senkou_span_a': senkou_span_a,
        'senkou_span_b': senkou_span_b,
        'chikou_span': chikou_span
    }

def volume_oscillator(volume: pd.Series, short_period: int = 5, long_period: int = 10) -> pd.Series:
    """
    Volume Oscillator - measures relationship between two volume moving averages.
    """
    short_ma = volume.rolling(short_period).mean()
    long_ma = volume.rolling(long_period).mean()
    
    vo = 100 * (short_ma - long_ma) / long_ma
    return vo

def chaikin_money_flow(high: pd.Series, low: pd.Series, close: pd.Series, 
                      volume: pd.Series, period: int = 20) -> pd.Series:
    """
    Chaikin Money Flow - measures money flow volume over specified period.
    Positive values indicate buying pressure, negative indicate selling pressure.
    """
    money_flow_multiplier = ((close - low) - (high - close)) / (high - low)
    money_flow_multiplier = money_flow_multiplier.fillna(0)
    
    money_flow_volume = money_flow_multiplier * volume
    cmf = money_flow_volume.rolling(period).sum() / volume.rolling(period).sum()
    
    return cmf

def on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    On-Balance Volume - relates volume to price change.
    Rising OBV suggests bullish sentiment, falling OBV suggests bearish.
    """
    obv = pd.Series(index=close.index, dtype=float)
    obv.iloc[0] = volume.iloc[0]
    
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    return obv

def aroon_indicator(high: pd.Series, low: pd.Series, period: int = 14) -> Tuple[pd.Series, pd.Series]:
    """
    Aroon Indicator - measures trend strength and potential reversals.
    
    Returns:
    - Aroon Up: Measures uptrend strength
    - Aroon Down: Measures downtrend strength
    """
    aroon_up = pd.Series(index=high.index, dtype=float)
    aroon_down = pd.Series(index=low.index, dtype=float)
    
    for i in range(period, len(high)):
        high_period = high.iloc[i-period+1:i+1]
        low_period = low.iloc[i-period+1:i+1]
        
        high_idx = high_period.idxmax()
        low_idx = low_period.idxmin()
        
        periods_since_high = period - 1 - (high_period.index.get_loc(high_idx))
        periods_since_low = period - 1 - (low_period.index.get_loc(low_idx))
        
        aroon_up.iloc[i] = ((period - periods_since_high) / period) * 100
        aroon_down.iloc[i] = ((period - periods_since_low) / period) * 100
    
    return aroon_up, aroon_down

def average_directional_index(high: pd.Series, low: pd.Series, close: pd.Series, 
                             period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Average Directional Index (ADX) - measures trend strength.
    
    Returns:
    - ADX: Trend strength (values above 25 indicate strong trend)
    - +DI: Positive directional indicator
    - -DI: Negative directional indicator
    """
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    high_diff = high - high.shift()
    low_diff = low.shift() - low
    
    plus_dm = pd.Series(0.0, index=high.index)
    minus_dm = pd.Series(0.0, index=low.index)
    
    plus_dm[(high_diff > low_diff) & (high_diff > 0)] = high_diff
    minus_dm[(low_diff > high_diff) & (low_diff > 0)] = low_diff
    
    # Smoothed True Range and Directional Movement
    atr = tr.rolling(period).mean()
    plus_dm_smooth = plus_dm.rolling(period).mean()
    minus_dm_smooth = minus_dm.rolling(period).mean()
    
    # Directional Indicators
    plus_di = 100 * (plus_dm_smooth / atr)
    minus_di = 100 * (minus_dm_smooth / atr)
    
    # ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(period).mean()
    
    return adx, plus_di, minus_di

def keltner_channels(high: pd.Series, low: pd.Series, close: pd.Series, 
                    period: int = 20, multiplier: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Keltner Channels - volatility-based envelope indicator.
    
    Returns:
    - Upper channel
    - Middle line (EMA)
    - Lower channel
    """
    typical_price = (high + low + close) / 3
    middle_line = typical_price.ewm(span=period).mean()
    
    # Average True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=period).mean()
    
    upper_channel = middle_line + (multiplier * atr)
    lower_channel = middle_line - (multiplier * atr)
    
    return upper_channel, middle_line, lower_channel

def price_channel(high: pd.Series, low: pd.Series, period: int = 20) -> Tuple[pd.Series, pd.Series]:
    """
    Price Channel (Donchian Channel) - breakout indicator.
    
    Returns:
    - Upper channel (highest high over period)
    - Lower channel (lowest low over period)
    """
    upper_channel = high.rolling(period).max()
    lower_channel = low.rolling(period).min()
    
    return upper_channel, lower_channel

def ultimate_oscillator(high: pd.Series, low: pd.Series, close: pd.Series,
                       period1: int = 7, period2: int = 14, period3: int = 28) -> pd.Series:
    """
    Ultimate Oscillator - momentum indicator using multiple timeframes.
    Reduces false signals by incorporating multiple time periods.
    """
    # True Range and Buying Pressure
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    bp = close - pd.concat([low, close.shift()], axis=1).min(axis=1)
    
    # Calculate averages for each period
    avg1 = bp.rolling(period1).sum() / tr.rolling(period1).sum()
    avg2 = bp.rolling(period2).sum() / tr.rolling(period2).sum()
    avg3 = bp.rolling(period3).sum() / tr.rolling(period3).sum()
    
    # Ultimate Oscillator calculation
    uo = 100 * ((4 * avg1) + (2 * avg2) + avg3) / (4 + 2 + 1)
    
    return uo

def vortex_indicator(high: pd.Series, low: pd.Series, close: pd.Series, 
                    period: int = 14) -> Tuple[pd.Series, pd.Series]:
    """
    Vortex Indicator - identifies trend changes and measures trend strength.
    
    Returns:
    - VI+: Positive vortex indicator
    - VI-: Negative vortex indicator
    """
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Vortex Movement
    vm_plus = abs(high - low.shift())
    vm_minus = abs(low - high.shift())
    
    # Vortex Indicators
    vi_plus = vm_plus.rolling(period).sum() / tr.rolling(period).sum()
    vi_minus = vm_minus.rolling(period).sum() / tr.rolling(period).sum()
    
    return vi_plus, vi_minus
