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
