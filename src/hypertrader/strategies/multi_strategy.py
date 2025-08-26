"""Collection of simple multi-strategy helpers.

The functions provided here serve as building blocks for arbitrage,
scalping and market-making strategies.  They are intentionally minimal
and operate on the best-effort basis, serving as examples of how to
extend the existing rule-based strategies in ``hypertrader``.
"""

from __future__ import annotations

from typing import Tuple

import ccxt


def arb_scalp(
    primary_ex: str = "binance",
    secondary_ex: str = "bybit",
    symbol: str = "BTC/USDT",
    threshold: float = 0.001,
    capital: float = 100.0,
    lev: int = 50,
) -> str:
    """Very small price-difference arbitrage helper.

    Fetches the latest ticker from two exchanges via :mod:`ccxt` and
    reports whether a buy-low/sell-high opportunity exists.  The
    function does not execute any trades; callers can use the returned
    string to decide their next action.
    """

    p1 = getattr(ccxt, primary_ex)().fetch_ticker(symbol)["last"]
    p2 = getattr(ccxt, secondary_ex)().fetch_ticker(symbol)["last"]
    if abs(p1 - p2) > threshold * p1:
        return "buy_low_sell_high" if p1 < p2 else "sell_high_buy_low"
    return "neutral"


def market_making(mid_price: float, spread: float = 0.001, lev: int = 20) -> Tuple[float, float]:
    """Return bid and ask quotes for a basic market-making strategy."""

    bid = mid_price * (1 - spread / 2)
    ask = mid_price * (1 + spread / 2)
    return bid, ask
