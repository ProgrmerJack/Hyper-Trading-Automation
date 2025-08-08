from hypertrader.execution.validators import validate_order


def test_validate_order_limits() -> None:
    market = {
        "limits": {
            "amount": {"min": 0.001, "step": 0.001},
            "cost": {"min": 10},
        }
    }
    assert not validate_order(20000, 0.0004, market)
    assert not validate_order(20000, 0.0015, market)
    assert validate_order(20000, 0.001, market)
