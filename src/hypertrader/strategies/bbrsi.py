"""
BBRSI Strategy
==============

This module implements a Bollinger Bands + RSI + ADX strategy for
cryptocurrency trading.  The logic here is inspired by the BBRSI
strategy used in the open‑source HyperLiquid trading bots.  The goal is
to provide a clean and configurable Python implementation that can be
backtested within the hypertrader framework.

The strategy evaluates the latest price data to determine when to
enter or exit long and short positions.  It combines three popular
technical indicators:

1. **Bollinger Bands** – measure volatility and identify relative
   high/low prices.  When the price crosses below the lower band, the
   market may be oversold; crossing above the upper band may signal an
   overbought condition.
2. **Relative Strength Index (RSI)** – identifies momentum extremes
   between 0 and 100.  Values below the oversold threshold suggest
   bullish reversal potential; values above the overbought threshold
   suggest bearish reversal potential.
3. **Average Directional Index (ADX)** – measures trend strength on
   a scale of 0–100.  Readings above a threshold (commonly 20–25)
   indicate a strong trend.  We require ADX to exceed a configurable
   threshold before taking trades.

Entry and exit rules:

* **Long entry** – price crosses below the lower Bollinger band and
  RSI is oversold while ADX exceeds the threshold.
* **Short entry** – price crosses above the upper Bollinger band and
  RSI is overbought while ADX exceeds the threshold.
* **Exit long** – price crosses back below the middle band or RSI
  rises above an extreme (e.g. 80).
* **Exit short** – price crosses back above the lower band or RSI
  falls below an extreme (e.g. 20).

Parameters like RSI periods, Bollinger periods/std dev and ADX
periods/thresholds are configurable via the constructor.  The
strategy exposes an ``update`` method compatible with the backtesting
engine: it consumes the latest price and returns a list of orders.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np

from ..indicators.technical import rsi as ta_rsi, bollinger_bands, adx as ta_adx


@dataclass
class BBRSIStrategy:
    """Bollinger Bands + RSI + ADX strategy.

    Attributes
    ----------
    symbol : str
        Trading pair, e.g. ``"BTC/USDT"``.  Stored for clarity but not
        currently used directly in the calculations.
    rsi_period : int
        Lookback window for RSI.
    rsi_overbought : float
        RSI value above which the market is considered overbought.
    rsi_oversold : float
        RSI value below which the market is considered oversold.
    bb_period : int
        Lookback window for Bollinger Bands.
    bb_std : float
        Standard deviation multiplier for Bollinger Bands.
    adx_period : int
        Lookback window for ADX.
    adx_threshold : float
        Minimum ADX required to take trades.
    profit_target : float
        Take‑profit expressed as a fraction (e.g. 0.05 = 5%).  If 0,
        no explicit take profit is set.

    Notes
    -----
    The ``update`` method accepts the current price and a history of
    OHLCV bars.  It computes indicators from the history and returns
    zero or more order tuples ``(side, price, qty)``.  In a backtest,
    ``price`` may be ``None`` to indicate market orders executed at the
    last trade price.  The strategy does not manage position size; it
    delegates quantity decisions to the backtester or risk manager.
    """

    symbol: str
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    bb_period: int = 20
    bb_std: float = 2.0
    adx_period: int = 14
    adx_threshold: float = 25.0
    profit_target: float = 0.03
    logger: Optional[any] = field(default=None, repr=False)

    # internal state
    last_signal: str = field(default="NONE", init=False)
    last_entry_price: Optional[float] = field(default=None, init=False)

    def update(self, price: float, history: List[dict]) -> List[Tuple[str, Optional[float], float]]:
        """Generate orders based on latest price and historical data.

        Parameters
        ----------
        price : float
            Latest traded price.  Used for exits and take‑profit levels.
        history : list of dict
            Recent OHLCV bars with keys ``open``, ``high``, ``low``,
            ``close``, ``volume``.  Must contain at least
            ``max(bb_period, rsi_period, adx_period)`` entries.  If
            insufficient data is provided, the method returns no
            orders.

        Returns
        -------
        list
            List of orders ``(side, price, qty)``.  The backtest
            engine interprets ``price=None`` as a market order.  By
            default this strategy returns either one entry order or
            one exit order.  Quantity is always 1 here – the caller
            should size trades appropriately.
        """
        # Check we have enough data
        min_bars = max(self.bb_period, self.rsi_period, self.adx_period) + 2
        if len(history) < min_bars:
            return []

        # Extract close prices for technical indicators
        closes = [bar["close"] for bar in history]
        highs = [bar["high"] for bar in history]
        lows = [bar["low"] for bar in history]

        # Compute indicators
        # RSI wrapper returns a single float for the latest value
        rsi_value = ta_rsi(closes, period=self.rsi_period)

        # Bollinger bands wrapper returns three floats: mid, upper, lower
        mid_band, upper_band, lower_band = bollinger_bands(closes, self.bb_period, self.bb_std)

        # ADX wrapper returns a single float
        adx_value = ta_adx(highs, lows, closes, period=self.adx_period)

        previous_close = closes[-2]
        current_close = closes[-1]

        # Entry conditions
        crossed_below_lower = previous_close >= lower_band and current_close < lower_band
        crossed_above_upper = previous_close <= upper_band and current_close > upper_band

        enter_long = crossed_below_lower and rsi_value < self.rsi_oversold and adx_value >= self.adx_threshold
        enter_short = crossed_above_upper and rsi_value > self.rsi_overbought and adx_value >= self.adx_threshold

        # Exit conditions
        crossed_under_middle = previous_close >= mid_band and current_close < mid_band
        crossed_under_lower = previous_close >= lower_band and current_close < lower_band
        rsi_exit_long = rsi_value > 80
        rsi_exit_short = rsi_value < 20

        orders: List[Tuple[str, Optional[float], float]] = []

        # Determine if we should exit
        if self.last_signal == "LONG":
            # Check take profit
            if self.profit_target and self.last_entry_price:
                target = self.last_entry_price * (1 + self.profit_target)
                if price >= target:
                    orders.append(("sell", None, 1.0))
                    self.last_signal = "NONE"
                    self.last_entry_price = None
                    return orders
            # Check exit conditions
            if crossed_under_middle or rsi_exit_long:
                orders.append(("sell", None, 1.0))
                self.last_signal = "NONE"
                self.last_entry_price = None
                return orders
        elif self.last_signal == "SHORT":
            if self.profit_target and self.last_entry_price:
                target = self.last_entry_price * (1 - self.profit_target)
                if price <= target:
                    orders.append(("buy", None, 1.0))
                    self.last_signal = "NONE"
                    self.last_entry_price = None
                    return orders
            # Exit conditions for short
            if crossed_under_lower or rsi_exit_short:
                orders.append(("buy", None, 1.0))
                self.last_signal = "NONE"
                self.last_entry_price = None
                return orders

        # Otherwise look for entries
        if self.last_signal == "NONE":
            if enter_long:
                orders.append(("buy", None, 1.0))
                self.last_signal = "LONG"
                self.last_entry_price = price
            elif enter_short:
                orders.append(("sell", None, 1.0))
                self.last_signal = "SHORT"
                self.last_entry_price = price
        return orders


__all__ = ["BBRSIStrategy"]