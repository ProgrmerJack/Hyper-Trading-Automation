"""
Event‑driven trading strategy (stub).

Event trading involves reacting to news releases, macroeconomic data,
social media posts or other external signals that can move markets.
High‑frequency event traders use low latency news feeds and natural
language processing to parse announcements within milliseconds and
generate trades before prices fully adjust.  This strategy can be
profitable when markets under‑react or when the trader can quickly
infer the direction of the move.

Implementing event trading in a production setting requires access
to proprietary data feeds, a robust parser and predictive models.
This module defines a placeholder class that documents the interface
and variables you would need.  The ``update`` method demonstrates
how an external event (e.g. sentiment score) might be used to
generate a buy or sell signal.  Users are encouraged to integrate
actual data sources and models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class EventTradingStrategy:
    """Stub for a news/event driven trading strategy.

    Parameters
    ----------
    symbol : str
        Trading pair to act on.
    sentiment_threshold : float, optional
        Threshold for positive or negative sentiment scores to
        trigger trades.  Scores above this value result in buys;
        below the negative of this value result in sells.
    position_size : float, optional
        Quantity to trade when an event occurs.
    """

    symbol: str
    sentiment_threshold: float = 0.5
    position_size: float = 1.0

    def update(self, sentiment_score: float) -> List[Tuple[str, str, float]]:
        """Generate orders based on an external sentiment score.

        This method demonstrates how a sentiment score (for example
        derived from a news headline or social media post) could be
        converted into a market order.  A positive score triggers a
        buy, a negative score triggers a sell, and values within the
        threshold result in no trade.

        Parameters
        ----------
        sentiment_score : float
            A scalar in [‑1, 1] representing bullish (positive) or
            bearish (negative) sentiment.

        Returns
        -------
        list of tuple
            A list of orders to execute, each as ``(side, symbol, quantity)``.
        """
        orders: List[Tuple[str, str, float]] = []
        if sentiment_score > self.sentiment_threshold:
            orders.append(("buy", self.symbol, self.position_size))
        elif sentiment_score < -self.sentiment_threshold:
            orders.append(("sell", self.symbol, self.position_size))
        return orders
