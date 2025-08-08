from hypertrader.risk.manager import RiskManager, RiskParams


def test_daily_loss_limit() -> None:
    rm = RiskManager(RiskParams(max_daily_loss=100, max_position=1000, fee_rate=0.001))
    rm.reset_day(1000)
    assert rm.check_order(950, 100, 0.002)
    assert not rm.check_order(800, 100, 0.002)


def test_exposure_and_fee_checks() -> None:
    rm = RiskManager(RiskParams(max_daily_loss=100, max_position=500, fee_rate=0.001))
    rm.reset_day(1000)
    assert not rm.check_order(1000, 600, 0.005)
    assert not rm.check_order(1000, 100, 0.0005)
    assert rm.check_order(1000, 100, 0.005)
