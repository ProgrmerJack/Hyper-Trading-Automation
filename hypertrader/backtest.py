"""Vectorized backtesting utilities using vectorbt."""

from __future__ import annotations

from typing import Callable, Tuple

import pandas as pd
import vectorbt as vbt


def advanced_backtest(
    data: pd.DataFrame,
    strategy_func: Callable[[pd.DataFrame], Tuple[pd.Series, pd.Series]],
    slippage_bps: float = 10.0,
    fee_bps: float = 5.0,
    leverage: float = 1.0,
):
    """Run a vectorized backtest on price data.

    Parameters
    ----------
    data : pd.DataFrame
        OHLCV data containing at least a ``close`` column.
    strategy_func : callable
        Function returning ``(entries, exits)`` boolean Series aligned with ``data``.
    slippage_bps : float, default 10.0
        Slippage per trade expressed in basis points.
    fee_bps : float, default 5.0
        Transaction cost per trade in basis points.
    leverage : float, default 1.0
        Multiplier applied to portfolio end value to approximate leverage.

    Returns
    -------
    dict
        Dictionary with portfolio statistics such as final value and Sharpe ratio.
    """
    entries, exits = strategy_func(data)
    pf = vbt.Portfolio.from_signals(
        close=data["close"],
        entries=entries,
        exits=exits,
        slippage=slippage_bps / 10000.0,
        fees=fee_bps / 10000.0,
    )
    stats = pf.stats()
    sharpe = pf.sharpe_ratio(freq="1D")
    end_value = float(stats["End Value"]) * leverage
    return {
        "final_value": end_value,
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(stats["Max Drawdown [%]"]),
    }


def exec_trade(pnl: float, side: str, qty: float, price: float, maker: bool = True, slip_bps: float = 5) -> float:
    fee_bps = -2 if maker else 10
    cost_bps = fee_bps + slip_bps
    return pnl - (abs(qty) * price * cost_bps / 10000)


__all__ = ["advanced_backtest", "exec_trade"]


def complex_backtest(data: pd.DataFrame, model) -> dict:
    """Backtest using prediction probabilities from a model.

    Parameters
    ----------
    data:
        DataFrame containing a ``close`` column and any features required by
        ``model``.
    model:
        Object implementing a ``predict`` method returning probabilities of
        upward movement. Values above ``0.5`` trigger long entries while values
        below ``0.5`` signal exits.

    Returns
    -------
    dict
        Dictionary containing Sharpe ratio and maximum drawdown statistics.
    """

    probs = model.predict(data)
    entries = probs > 0.5
    exits = probs < 0.5
    pf = vbt.Portfolio.from_signals(data["close"], entries, exits)
    pf = pf.apply_slippage(0.001).apply_fees(0.0005)
    stats = pf.stats()
    return {
        "sharpe_ratio": float(stats["Sharpe Ratio"]),
        "max_drawdown": float(stats["Max Drawdown [%]"]),
    }


__all__.append("complex_backtest")

