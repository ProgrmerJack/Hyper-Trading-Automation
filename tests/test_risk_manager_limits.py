from hypertrader.risk.manager import RiskManager, RiskParams


def test_daily_loss_limit() -> None:
    rm = RiskManager(
        RiskParams(max_daily_loss=100, max_position=1000, fee_rate=0.001, slippage=0.0005)
    )
    rm.reset_day(1000)
    assert rm.check_order(950, "BTC-USD", 100, 0.002)
    assert not rm.check_order(800, "BTC-USD", 100, 0.002)


def test_exposure_fee_slippage_and_symbol_limit() -> None:
    params = RiskParams(
        max_daily_loss=100,
        max_position=500,
        fee_rate=0.001,
        slippage=0.0005,
        symbol_limits={"ETH-USD": 300},
    )
    rm = RiskManager(params)
    rm.reset_day(1000)
    assert not rm.check_order(1000, "ETH-USD", 600, 0.005)
    assert not rm.check_order(1000, "ETH-USD", 200, 0.001)
    assert not rm.check_order(1000, "ETH-USD", 400, 0.005)
    assert rm.check_order(1000, "BTC-USD", 200, 0.005)
