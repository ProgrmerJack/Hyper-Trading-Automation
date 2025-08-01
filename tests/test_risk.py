from hypertrader.utils.risk import calculate_position_size


def test_calculate_position_size():
    volume = calculate_position_size(10000, 2, 100, 95)
    # risk is 2% of 10000 = 200, stop distance=5 => volume=40
    assert round(volume, 2) == 40
