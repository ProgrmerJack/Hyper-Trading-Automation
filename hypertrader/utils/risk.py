"""Risk management utilities."""
from __future__ import annotations


def calculate_position_size(account_balance: float, risk_percent: float, entry_price: float, stop_loss_price: float) -> float:
    """Calculate position size based on risk percentage and stop loss distance."""
    if risk_percent <= 0 or account_balance <= 0:
        raise ValueError("Account balance and risk_percent must be positive")
    risk_amount = account_balance * (risk_percent / 100)
    stop_distance = abs(entry_price - stop_loss_price)
    if stop_distance == 0:
        raise ValueError("Stop distance cannot be zero")
    volume = risk_amount / stop_distance
    return volume


def trailing_stop(entry_price: float, current_price: float, atr: float, multiplier: float = 2.0) -> float:
    """Compute a trailing stop level based on ATR.

    Parameters
    ----------
    entry_price : float
        The price at which the position was opened.
    current_price : float
        Latest market price.
    atr : float
        Average true range value for volatility scaling.
    multiplier : float, default 2.0
        ATR multiplier determining stop distance.

    Returns
    -------
    float
        The new stop level that trails price movements.
    """
    stop_distance = atr * multiplier
    if current_price >= entry_price:
        return current_price - stop_distance
    return current_price + stop_distance


def drawdown_throttle(portfolio_value: float, peak_value: float, max_drawdown: float = 0.05) -> float:
    """Scale position allocation based on current drawdown.

    Returns an allocation factor between ``0.1`` and ``1.0`` that reduces
    exposure as drawdown approaches ``max_drawdown``.
    """
    if peak_value <= 0:
        raise ValueError("peak_value must be positive")
    current_dd = (peak_value - portfolio_value) / peak_value
    if current_dd <= 0:
        return 1.0
    alloc = 1 - (current_dd / max_drawdown)
    return max(0.1, alloc)


def kill_switch(current_drawdown: float, max_drawdown: float = 0.1) -> bool:
    """Determine whether trading should be halted due to drawdown breach."""
    return current_drawdown > max_drawdown


def dynamic_leverage(
    capital: float,
    risk_percent: float = 1.0,
    volatility: float = 0.02,
    min_leverage: float = 10.0,
    max_leverage: float = 100.0,
) -> float:
    """Determine leverage based on risk tolerance and market volatility.

    Parameters
    ----------
    capital : float
        Current trading capital.
    risk_percent : float, default 1.0
        Percentage of capital willing to risk on a trade.
    volatility : float, default 0.02
        Recent volatility estimate expressed as a decimal (e.g. ``0.02`` for 2%).
    min_leverage : float, default 10.0
        Minimum leverage allowed by the broker.
    max_leverage : float, default 100.0
        Maximum leverage allowed by the broker.

    Returns
    -------
    float
        Suggested leverage constrained between ``min_leverage`` and ``max_leverage``.
    """
    if any(x <= 0 for x in (capital, risk_percent, volatility)):
        raise ValueError("capital, risk_percent and volatility must be positive")

    max_loss = capital * (risk_percent / 100)
    lev = max_loss / (volatility * capital)
    return max(min_leverage, min(max_leverage, lev))


def compound_capital(capital: float, daily_return: float) -> float:
    """Compound trading capital by a daily return.

    Parameters
    ----------
    capital : float
        Current capital.
    daily_return : float
        Daily return expressed as decimal (e.g. ``0.02`` for 2%).

    Returns
    -------
    float
        Updated capital after applying the return.
    """
    return capital * (1 + daily_return)


def volatility_scaled_stop(
    entry_price: float,
    vix: float,
    base_percent: float = 0.01,
    long: bool = True,
) -> float:
    """Create a stop level scaled by market volatility.

    A higher volatility index widens the stop distance to reduce
    whipsaws during turbulent periods.
    """
    if entry_price <= 0 or vix < 0:
        raise ValueError("entry_price must be positive and vix non-negative")

    adjust = 1 + (vix / 100.0)
    distance = entry_price * base_percent * adjust
    return entry_price - distance if long else entry_price + distance


__all__ = [
    "calculate_position_size",
    "trailing_stop",
    "drawdown_throttle",
    "kill_switch",
    "dynamic_leverage",
    "compound_capital",
    "volatility_scaled_stop",
]
