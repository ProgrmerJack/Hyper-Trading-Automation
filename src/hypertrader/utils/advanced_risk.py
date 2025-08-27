"""Advanced risk management utilities.

This module provides an `AdvancedRiskManager` class that implements
several sophisticated position sizing techniques inspired by
professional high‑frequency trading systems.  It supports
volatility‑adjusted risk, Kelly criterion sizing, pyramiding,
anti‑martingale behaviour, and tracking of drawdowns and win rates.

The goal of this class is to centralise the logic around how much
capital to allocate to each trade given the current account equity
and market conditions.  It is intended to be used alongside the
strategy modules in :mod:`hypertrader.strategies` and the
execution engine in :mod:`hypertrader.bot`.

Usage example
-------------

```
from hypertrader.utils.advanced_risk import AdvancedRiskManager

# Initialise with starting equity of 1000 USDT
rm = AdvancedRiskManager(initial_equity=1000.0,
                         max_risk_per_trade=0.02,
                         max_drawdown=0.2,
                         volatility_period=14,
                         pyramiding=True,
                         anti_martingale=True,
                         kelly_fraction=0.5)

# Update market data on each new tick
rm.update_market(price)

# When a trade is closed, record its profit or loss
rm.record_trade(pnl)

# Determine the next position size (in units) at current price
qty = rm.calculate_position_size(price)

```

This risk manager maintains internal state about equity, drawdowns,
market volatility and trading performance in order to adapt its
position sizing rules.  See the docstrings of individual methods for
more details.
"""

from __future__ import annotations

import math
from typing import Deque, List, Optional
from collections import deque


class AdvancedRiskManager:
    """Manage risk and position sizing for a trading strategy.

    Parameters
    ----------
    initial_equity : float
        Starting capital of the trading account.  This value is used
        to compute drawdowns and maximum allowed risk per trade.
    max_risk_per_trade : float, optional
        Fraction of current equity to risk on any single trade.  For
        example, ``0.02`` means a maximum of 2% of equity.  Defaults
        to ``0.01`` (1%).
    max_drawdown : float, optional
        Maximum allowable drawdown (peak to trough) before trading is
        halted or positions are reduced.  Expressed as a fraction of
        peak equity (e.g. ``0.2`` for 20%).  Defaults to ``0.2``.
    volatility_period : int, optional
        Number of periods over which to estimate volatility.  This
        determines the responsiveness of the volatility adjustment.
        Defaults to ``14``.
    pyramiding : bool, optional
        If ``True``, the manager will allow incremental position
        increases when a trade is winning (adding to the position as
        it becomes profitable).  This is sometimes called scaling in.
        Defaults to ``False``.
    anti_martingale : bool, optional
        If ``True``, the manager will reduce position size after
        losing trades (opposite of a martingale).  This helps
        preserve capital during drawdowns.  Defaults to ``False``.
    kelly_fraction : float, optional
        Fraction of the Kelly bet to deploy when computing position
        sizes.  ``0.0`` disables Kelly sizing.  Values should be
        between 0 and 1.  Defaults to ``0.0``.
    """

    def __init__(
        self,
        initial_equity: float,
        max_risk_per_trade: float = 0.01,
        max_drawdown: float = 0.2,
        volatility_period: int = 14,
        pyramiding: bool = False,
        anti_martingale: bool = False,
        kelly_fraction: float = 0.0,
    ) -> None:
        self.initial_equity = float(initial_equity)
        self.equity = float(initial_equity)
        self.peak_equity = float(initial_equity)
        self.max_drawdown = float(max_drawdown)
        self.max_risk_per_trade = float(max_risk_per_trade)
        self.volatility_period = int(volatility_period)
        self.pyramiding = pyramiding
        self.anti_martingale = anti_martingale
        self.kelly_fraction = max(0.0, min(1.0, kelly_fraction))

        # Market history for volatility estimation
        self._price_history: Deque[float] = deque(maxlen=volatility_period + 1)
        self.volatility: float = 0.0

        # Performance tracking
        self.trades: List[float] = []
        self.wins = 0
        self.total_trades = 0
        self.consecutive_wins: int = 0
        self.consecutive_losses: int = 0

    @property
    def drawdown(self) -> float:
        """Current drawdown as a fraction of the peak equity."""
        if self.peak_equity == 0:
            return 0.0
        return max(0.0, (self.peak_equity - self.equity) / self.peak_equity)

    @property
    def win_rate(self) -> float:
        """Estimated win rate based on trade history.

        Returns the fraction of winning trades over the total number of
        completed trades.  If no trades have been recorded, returns
        ``0.5`` as a neutral default.
        """
        return self.wins / self.total_trades if self.total_trades > 0 else 0.0

    def update_market(self, price: float) -> None:
        """Update the market price history and compute volatility.

        The volatility estimate is computed as the standard deviation
        of log returns over the last ``volatility_period`` observations.
        If fewer than two prices are available, volatility remains
        unchanged.

        Parameters
        ----------
        price : float
            Latest market price to append to the history.
        """
        self._price_history.append(float(price))
        if len(self._price_history) > 1:
            # Compute logarithmic returns
            prices = list(self._price_history)
            returns = []
            for i in range(1, len(prices)):
                prev = prices[i - 1]
                cur = prices[i]
                if prev > 0:
                    returns.append(math.log(cur / prev))
            if len(returns) > 1:
                mean_ret = sum(returns) / len(returns)
                var = sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1)
                self.volatility = math.sqrt(var)

    def update_equity(self, real_equity: float) -> None:
        """Update equity from real account balance"""
        self.equity = real_equity
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

    def record_trade(self, real_pnl: float) -> None:
        """Record real trade P&L from exchange"""
        self.trades.append(float(real_pnl))
        # Update win rate and other metrics based on real P&L
        if real_pnl > 0:
            self.wins += 1
        self.total_trades += 1
        self.win_rate = self.wins / self.total_trades if self.total_trades > 0 else 0.0
        if real_pnl > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        elif real_pnl < 0:
            self.consecutive_losses += 1
            self.consecutive_wins = 0

    def _volatility_scaler(self, price: float) -> float:
        """Compute a size scaler based on current volatility.

        The scaler is proportional to the inverse of volatility times
        price, ensuring that size decreases when volatility increases.
        If volatility is zero (e.g. at start), returns 1.0.
        """
        if self.volatility > 0 and price > 0:
            # Use annualised volatility estimate (assume 365 periods) for scaling
            vol = self.volatility * math.sqrt(365)
            return 1.0 / (vol * price)
        return 1.0

    def _kelly_size(self, risk_reward: float = 2.0) -> float:
        """Compute the Kelly optimal fraction.

        Uses the current win rate estimate and an assumed risk‑reward
        ratio to determine the optimal fraction of capital to risk.  If
        the calculated Kelly fraction is negative (i.e., expected
        value is negative), returns zero.  The fraction is scaled by
        ``kelly_fraction`` to reduce leverage.

        Parameters
        ----------
        risk_reward : float, optional
            Assumed ratio of average win size to average loss size.  A
            value of 2.0 means wins are twice as big as losses on
            average.

        Returns
        -------
        float
            Kelly fraction to apply to the current equity.
        """
        if self.kelly_fraction <= 0.0:
            return self.max_risk_per_trade
        p = self.win_rate
        b = risk_reward
        # Kelly formula: f* = p - (1 - p) / b
        f_opt = p - (1.0 - p) / b
        if f_opt < 0:
            return 0.0
        return f_opt * self.kelly_fraction

    def calculate_position_size(self, price: float, risk_reward: float = 2.0) -> float:
        """Determine the position size (notional value) for the next trade.

        This method combines the configured maximum risk per trade,
        volatility adjustment, Kelly sizing, and optional pyramiding
        or anti‑martingale rules to produce a notional exposure.

        Parameters
        ----------
        price : float
            Current market price.  Used to convert monetary risk into
            position size units.
        risk_reward : float, optional
            Risk–reward ratio assumed for the Kelly calculation.  Only
            used if ``kelly_fraction`` > 0.

        Returns
        -------
        float
            Position size (number of units) to trade.  This can be
            positive (buy) or negative (sell); the sign should be
            applied by the strategy depending on signal direction.
        """
        if price <= 0 or self.equity <= 0:
            return 0.0

        # Base risk fraction (max risk per trade or Kelly fraction)
        risk_fraction = max(self.max_risk_per_trade, self._kelly_size(risk_reward))

        # Reduce risk if drawdown exceeds threshold
        if self.drawdown >= self.max_drawdown:
            risk_fraction *= 0.5

        # Volatility scaling: larger volatility -> smaller position
        vol_scale = self._volatility_scaler(price)

        # Pyramiding: increase position on winning streak
        pyramid_factor = 1.0
        if self.pyramiding and self.consecutive_wins > 0:
            pyramid_factor = 1.0 + 0.5 * self.consecutive_wins

        # Anti‑martingale: decrease position on losing streak
        martingale_factor = 1.0
        if self.anti_martingale and self.consecutive_losses > 0:
            martingale_factor = 1.0 / (1.0 + self.consecutive_losses)

        # Final position size in notional (monetary) terms
        notional = self.equity * risk_fraction * vol_scale * pyramid_factor * martingale_factor
        # Convert notional to quantity (units) by dividing by price
        qty = notional / price
        return qty