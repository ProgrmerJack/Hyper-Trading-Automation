"""
Statistical arbitrage strategy for pairs trading or cross‑exchange
mispricing.

This strategy monitors the price spread between two related assets or
markets and enters positions when the spread deviates significantly
from its historical mean.  The canonical example is pairs trading
where the price difference between two co‑integrated stocks reverts
to its long‑term mean; in crypto, common pairs include BTC vs ETH or
an instrument on two different exchanges.  When the spread
temporarily widens or narrows, the strategy simultaneously buys one
instrument and sells the other, betting on mean reversion.

Implementation details
----------------------

* The strategy maintains a rolling window of recent spreads.
* At each step it computes the z‑score of the current spread
  relative to the window mean and standard deviation.
* If the z‑score exceeds ``entry_threshold``, the strategy opens a
  short position on ``symbol_a`` and a long position on ``symbol_b``;
  if the z‑score is below ``-entry_threshold``, it does the opposite.
* The strategy exits (closes both legs) when the z‑score returns
  within ``exit_threshold`` of zero.

This approach does not account for execution latency, market impact
or order book depth.  It assumes that trades can be executed at
mid prices and that capital is sufficient to maintain the hedge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass
class StatisticalArbitrageStrategy:
    """Simple pairs trading strategy based on z‑score of price spreads.

    Parameters
    ----------
    symbol_a : str
        First instrument to trade (e.g., BTC/USDT on exchange A).
    symbol_b : str
        Second instrument (e.g., BTC/USDT on exchange B or a related
        asset such as ETH/USDT).  The strategy will buy one and sell
        the other depending on the direction of the mispricing.
    window : int
        Number of historical observations to use for computing the
        mean and standard deviation of the spread.
    entry_threshold : float
        Z‑score threshold to open a trade.  The absolute value of
        z‑score must exceed this number to initiate a position.
    exit_threshold : float
        Z‑score threshold to close a trade.  When the absolute
        z‑score falls below this number, open positions are closed.
    position_size : float
        Nominal size to trade on each leg.  In practice this should be
        adjusted based on risk limits and asset volatility.
    """

    symbol_a: str
    symbol_b: str
    window: int = 100
    entry_threshold: float = 2.0
    exit_threshold: float = 0.5
    position_size: float = 1.0
    # internal state
    spreads: List[float] = field(default_factory=list, init=False)
    in_position: bool = field(default=False, init=False)
    # +1 means long A / short B; -1 means short A / long B
    position_direction: int = field(default=0, init=False)

    def update(self, price_a: float, price_b: float) -> List[Tuple[str, str, float]]:
        """Update the strategy with new prices and return orders."""
        # Compute spread (simple difference).  More sophisticated
        # strategies might use a hedge ratio from regression.
        spread = price_a - price_b
        self.spreads.append(spread)
        if len(self.spreads) < self.window:
            # Not enough data yet
            return []
        # Trim history to window length
        if len(self.spreads) > self.window:
            self.spreads.pop(0)
        spreads_array = np.array(self.spreads)
        mean = float(np.mean(spreads_array))
        std = float(np.std(spreads_array))
        if std == 0.0:
            zscore = 0.0
        else:
            zscore = (spread - mean) / std

        orders: List[Tuple[str, str, float]] = []
        # If not currently in a position, check for entry
        if not self.in_position:
            if zscore > self.entry_threshold:
                # Spread too high: short A, long B
                orders.append(("sell", self.symbol_a, self.position_size))
                orders.append(("buy", self.symbol_b, self.position_size))
                self.in_position = True
                self.position_direction = -1
            elif zscore < -self.entry_threshold:
                # Spread too low: long A, short B
                orders.append(("buy", self.symbol_a, self.position_size))
                orders.append(("sell", self.symbol_b, self.position_size))
                self.in_position = True
                self.position_direction = 1
        else:
            # Already in a position; check for exit
            if abs(zscore) < self.exit_threshold:
                # Close both legs
                if self.position_direction == 1:
                    orders.append(("sell", self.symbol_a, self.position_size))
                    orders.append(("buy", self.symbol_b, self.position_size))
                else:
                    orders.append(("buy", self.symbol_a, self.position_size))
                    orders.append(("sell", self.symbol_b, self.position_size))
                self.in_position = False
                self.position_direction = 0
        return orders


@dataclass
class PairStatArb:
    """Minimal pairs trading strategy using z-score."""
    lookback: int = 100
    entry: float = 2.0
    exit: float = 0.5
    
    def update(self, a: pd.Series, b: pd.Series) -> Tuple[int, float]:
        """Update with price series, return (signal, z_score)."""
        if len(a) < self.lookback or len(b) < self.lookback:
            return 0, 0.0
        spread = a.tail(self.lookback).values - b.tail(self.lookback).values
        z = (spread[-1] - np.mean(spread)) / (np.std(spread) + 1e-9)
        if z > self.entry:
            return -1, float(z)  # Short A, Long B
        if z < -self.entry:
            return 1, float(z)   # Long A, Short B
        if abs(z) < self.exit:
            return 0, float(z)   # Close position
        return 0, float(z)


def pairs_signal(price_a: pd.Series, price_b: pd.Series, 
                 lookback: int = 100, entry_thresh: float = 2.0, 
                 exit_thresh: float = 0.5) -> Tuple[int, float]:
    """Generate pairs trading signal from price series."""
    if len(price_a) < lookback or len(price_b) < lookback:
        return 0, 0.0
    spread = price_a.tail(lookback).values - price_b.tail(lookback).values
    z = (spread[-1] - np.mean(spread)) / (np.std(spread) + 1e-9)
    if z > entry_thresh:
        return -1, float(z)
    elif z < -entry_thresh:
        return 1, float(z)
    elif abs(z) < exit_thresh:
        return 0, float(z)
    return 0, float(z)
