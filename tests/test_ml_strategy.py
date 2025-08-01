import pandas as pd

from hypertrader.strategies.ml_strategy import extract_features, train_model, ml_signal


def test_train_and_predict():
    df = pd.DataFrame({
        'open': [1,2,3,4,5,6,7,8,9,10],
        'high': [1,2,3,4,5,6,7,8,9,10],
        'low': [1,2,3,4,5,6,7,8,9,10],
        'close': [1,2,3,2,3,4,5,6,5,6],
        'volume': [1]*10,
    }, index=pd.date_range('2024-01-01', periods=10, freq='H'))
    model = train_model(df)
    sig = ml_signal(model, df)
    assert sig.action in {'BUY','SELL','HOLD'}
    assert 0.0 <= sig.probability <= 1.0
