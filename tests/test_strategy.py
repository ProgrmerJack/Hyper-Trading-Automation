import pandas as pd

from hypertrader.strategies.indicator_signals import generate_signal


def test_generate_signal_runs():
    data = pd.DataFrame({
        'open': [1]*300,
        'high': [1.2]*300,
        'low': [0.8]*300,
        'close': [1]*200 + [2]*100,
    })
    sig = generate_signal(data, sentiment_score=0.5, macro_score=0.5)
    assert sig.action in {'BUY', 'SELL', 'HOLD'}


def test_generate_signal_hold_when_insufficient_data():
    data = pd.DataFrame({
        'open': list(range(100)),
        'high': [x + 0.5 for x in range(100)],
        'low': [x - 0.5 for x in range(100)],
        'close': list(range(100)),
    })
    sig = generate_signal(data, sentiment_score=0, macro_score=0)
    assert sig.action == 'HOLD'


def test_generate_signal_with_onchain_and_skew():
    import numpy as np
    close = [1]*200 + list(np.linspace(1.0, 1.1, 86)) + [1.09, 1.1]*7
    data = pd.DataFrame({'close': close})
    sig = generate_signal(
        data,
        sentiment_score=0.5,
        macro_score=0.5,
        onchain_score=2.0,
        book_skew=0.3,
    )
    assert sig.action == 'BUY'


def test_generate_signal_sell_with_onchain_and_skew():
    import numpy as np
    close = [2]*200 + list(np.linspace(2.0, 1.9, 86)) + [1.91, 1.9]*7
    data = pd.DataFrame({'close': close})
    sig = generate_signal(
        data,
        sentiment_score=-0.5,
        macro_score=-0.5,
        onchain_score=-2.0,
        book_skew=-0.3,
    )
    assert sig.action == 'SELL'
