import pandas as pd
import pytest

from hypertrader.data.fetch_data import fetch_ohlcv, fetch_yahoo_ohlcv


def test_fetch_ohlcv(monkeypatch):
    class DummyExchange:
        def fetch_ohlcv(self, symbol, timeframe='1h', since=None, limit=1000):
            return [
                [0, 1, 2, 0, 1, 10],
                [3600000, 1, 2, 0, 1, 10],
            ]
    monkeypatch.setattr('ccxt.binance', DummyExchange)
    df = fetch_ohlcv('binance', 'BTC/USDT')
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ['open', 'high', 'low', 'close', 'volume']
    assert len(df) == 2


def test_fetch_yahoo_ohlcv(monkeypatch):
    def dummy_download(symbol, period='7d', interval='1h', progress=False):
        idx = pd.date_range('2024-01-01', periods=2, freq='H')
        data = {
            'Open': [1, 1],
            'High': [2, 2],
            'Low': [0, 0],
            'Close': [1, 1],
            'Volume': [10, 20],
        }
        return pd.DataFrame(data, index=idx)

    monkeypatch.setattr('yfinance.download', dummy_download)
    df = fetch_yahoo_ohlcv('BTC-USD')
    assert list(df.columns) == ['open', 'high', 'low', 'close', 'volume']
    assert len(df) == 2
