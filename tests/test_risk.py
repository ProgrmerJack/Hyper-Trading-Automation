from hypertrader.utils.risk import (
    calculate_position_size,
    drawdown_throttle,
    kill_switch,
    trailing_stop,
    dynamic_leverage,
    compound_capital,
    volatility_scaled_stop,
    quantum_leverage_modifier,
    cap_position_value,
)


def test_calculate_position_size():
    volume = calculate_position_size(10000, 2, 100, 95)
    # risk is 2% of 10000 = 200, stop distance=5 => volume=40
    assert round(volume, 2) == 40


def test_position_size_respects_risk_limit():
    volume = calculate_position_size(100, 5, 50, 45)
    # risk per unit is 5, exposure should be <=5 dollars
    assert volume * (50 - 45) <= 5


def test_trailing_stop():
    stop = trailing_stop(100, 110, atr=5, multiplier=2)
    assert stop == 110 - 10


def test_drawdown_throttle():
    alloc = drawdown_throttle(9500, 10000, max_drawdown=0.1)
    assert round(alloc, 2) == 0.5


def test_kill_switch():
    assert kill_switch(0.11, max_drawdown=0.1)
    assert not kill_switch(0.05, max_drawdown=0.1)


def test_dynamic_leverage():
    lev = dynamic_leverage(100, risk_percent=1, volatility=0.0005)
    assert 1 <= lev <= 5
    assert lev > 1  # should scale above minimum with low vol


def test_compound_and_vol_stop():
    capital = compound_capital(100, 0.02)
    assert round(capital, 2) == 102
    stop = volatility_scaled_stop(100, vix=20)
    assert round(stop, 2) == 98.8


def test_quantum_leverage_modifier_fallback():
    factor = quantum_leverage_modifier([0.1, 0.2])
    assert factor == 1.0


def test_cap_position_value_limits_notional():
    capped = cap_position_value(10, 50, 100, 3)
    assert capped == 6
