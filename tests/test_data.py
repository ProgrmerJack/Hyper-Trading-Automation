import pandas as pd
import pytest

from hypertrader.data.fetch_data import fetch_ohlcv, fetch_yahoo_ohlcv, fetch_order_book
from hypertrader.data.macro import fetch_cardboard_production


def test_fetch_ohlcv(monkeypatch):
    class DummyExchange:
        def fetch_ohlcv(self, symbol, timeframe='1h', since=None, limit=1000):
            return [
                [0, 1, 2, 0, 1, 10],
                [3600000, 1, 2, 0, 1, 10],
            ]
        def close(self):
            pass
    monkeypatch.setattr('ccxt.binance', DummyExchange)
    df = fetch_ohlcv('binance', 'BTC/USDT')
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ['open', 'high', 'low', 'close', 'volume']
    assert len(df) == 2


def test_fetch_ohlcv_with_fallback(monkeypatch):
    class PrimaryExchange:
        def fetch_ohlcv(self, symbol, timeframe='1h', since=None, limit=1000):
            raise RuntimeError("primary fail")
        def close(self):
            pass

    class FallbackExchange:
        def fetch_ohlcv(self, symbol, timeframe='1h', since=None, limit=1000):
            return [[0, 1, 1, 1, 1, 1]]
        def close(self):
            pass

    monkeypatch.setattr('ccxt.binance', PrimaryExchange)
    monkeypatch.setattr('ccxt.coinbase', FallbackExchange)

    df = fetch_ohlcv('binance', 'BTC/USDT', fallback='coinbase')
    assert not df.empty


def test_fetch_yahoo_ohlcv(monkeypatch):
    def dummy_download(symbol, period='7d', interval='1h', progress=False):
        idx = pd.date_range('2024-01-01', periods=2, freq='h')
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


def test_fetch_cardboard_production(monkeypatch):
    def dummy_fetch(series_id, api_key):
        return pd.Series([1, 2, 3])

    monkeypatch.setattr('hypertrader.data.macro.fetch_fred_series', dummy_fetch)
    series = fetch_cardboard_production('dummy')
    assert list(series) == [1, 2, 3]


def test_fetch_order_book(monkeypatch):
    class DummyExchange:
        def fetch_order_book(self, symbol, limit=5):
            return {"bids": [[1, 2]], "asks": [[1.1, 3]]}
        def close(self):
            pass

    monkeypatch.setattr('ccxt.binance', DummyExchange)
    book = fetch_order_book('binance', 'BTC/USDT')
    assert 'bids' in book and 'asks' in book
