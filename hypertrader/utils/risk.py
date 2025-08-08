"""Risk management utilities."""
from __future__ import annotations

from importlib import import_module
from typing import Sequence

import numpy as np


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


def cap_position_value(
    volume: float,
    price: float,
    account_balance: float,
    max_exposure: float,
) -> float:
    """Clamp position size so notional does not exceed allowed exposure.

    Parameters
    ----------
    volume : float
        Position size in units of the base asset.
    price : float
        Current asset price.
    account_balance : float
        Current account equity.
    max_exposure : float
        Maximum allowable exposure expressed as a multiple of equity
        (e.g. ``3`` for ``3x`` the account balance).

    Returns
    -------
    float
        Adjusted volume respecting the exposure cap.
    """
    if any(x < 0 for x in (volume, price, account_balance, max_exposure)):
        raise ValueError("inputs must be non-negative")
    max_value = account_balance * max_exposure
    value = volume * price
    if value > max_value and price > 0:
        return max_value / price
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


def ai_var(returns: Sequence[float], confidence: float = 0.95) -> float:
    """Estimate Value-at-Risk (VaR) using historical simulation.

    Parameters
    ----------
    returns : Sequence[float]
        Historical portfolio returns.
    confidence : float, default ``0.95``
        Confidence level for VaR calculation.

    Returns
    -------
    float
        VaR value expressed as a positive number. Higher values indicate
        greater expected loss.
    """
    if not returns:
        raise ValueError("returns cannot be empty")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    sorted_returns = np.sort(np.asarray(returns))
    index = int(len(sorted_returns) * (1 - confidence))
    index = max(0, min(index, len(sorted_returns) - 1))
    return float(-sorted_returns[index])


def drl_throttle(state: tuple[float, float]) -> float:
    """Adaptive throttle factor using a reinforcement learning agent.

    The function attempts to load ``stable_baselines3`` to train a minimal
    agent. If unavailable, it falls back to a heuristic based on drawdown and
    volatility.

    Parameters
    ----------
    state : tuple[float, float]
        A tuple of ``(drawdown, volatility)``.

    Returns
    -------
    float
        Throttle factor between ``0.1`` and ``1``.
    """
    drawdown, vol = state
    try:  # pragma: no cover - optional dependency
        sb3 = import_module("stable_baselines3")
        gym = import_module("gym")
        # Simple environment describing risk state
        class RiskEnv(gym.Env):
            metadata = {"render.modes": []}

            def __init__(self):
                super().__init__()
                self.observation_space = gym.spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
                self.action_space = gym.spaces.Box(low=0.1, high=1.0, shape=(1,), dtype=np.float32)

            def reset(self, *, seed=None, options=None):  # type: ignore[override]
                return np.zeros(2, dtype=np.float32), {}

            def step(self, action):
                # reward encourages lower throttle when risk is high
                reward = 1 - (drawdown + vol) * action[0]
                done = True
                return np.array([drawdown, vol], dtype=np.float32), reward, done, False, {}

        model = sb3.PPO("MlpPolicy", RiskEnv(), verbose=0)
        model.learn(100)
        action, _ = model.predict(np.array([drawdown, vol]))
        return float(action[0])
    except Exception:
        # fallback heuristic
        return float(max(0.1, 1 - drawdown - vol))


def quantum_leverage_modifier(
    features: Sequence[float], n_qubits: int = 4, shots: int = 1000
) -> float:
    """Estimate a leverage modifier using a quantum circuit.

    The function attempts to leverage :mod:`qiskit` to derive a value between
    ``0`` and ``1`` that can be used to scale leverage. If :mod:`qiskit` is not
    installed, a neutral factor of ``1.0`` is returned.

    Parameters
    ----------
    features : Sequence[float]
        Arbitrary feature vector representing market state.
    n_qubits : int, default ``4``
        Number of qubits for the circuit. Higher values increase resolution.
    shots : int, default ``1000``
        Number of circuit executions for sampling.

    Returns
    -------
    float
        A scaling factor in the range ``[0, 1]``.
    """

    if not features:
        return 1.0
    try:  # pragma: no cover - optional dependency
        from qiskit import Aer, execute
        from qiskit.circuit.library import EfficientSU2
    except Exception:
        return 1.0

    qc = EfficientSU2(n_qubits)
    backend = Aer.get_backend("aer_simulator")
    result = execute(qc, backend, shots=shots).result()
    counts = result.get_counts()
    optimal = max(counts, key=counts.get)
    return int(optimal, 2) / (2 ** n_qubits - 1)


def shap_explain(model, features):
    """Compute SHAP values for a model prediction.

    Returns ``None`` if the :mod:`shap` library is not installed."""
    try:  # pragma: no cover - optional dependency
        shap = import_module("shap")
    except Exception:  # shap not installed
        return None
    explainer = shap.Explainer(model)
    return explainer(features)


__all__ = [
    "calculate_position_size",
    "trailing_stop",
    "drawdown_throttle",
    "kill_switch",
    "dynamic_leverage",
    "cap_position_value",
    "compound_capital",
    "volatility_scaled_stop",
    "ai_var",
    "drl_throttle",
    "quantum_leverage_modifier",
    "shap_explain",
]
