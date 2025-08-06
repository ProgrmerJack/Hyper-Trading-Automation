import json
from pathlib import Path
import pandas as pd

from hypertrader.bot import run


def test_run_creates_signal(monkeypatch, tmp_path):
    def dummy_download(symbol, period='7d', interval='1h', progress=False):
        idx = pd.date_range('2024-01-01', periods=200, freq='h')
        data = {
            'Open': [1]*200,
            'High': [1]*200,
            'Low': [1]*200,
            'Close': [1]*200,
            'Volume': [1]*200,
        }
        return pd.DataFrame(data, index=idx)

    monkeypatch.setattr('yfinance.download', dummy_download)
    monkeypatch.setattr('hypertrader.bot.fetch_news_headlines', lambda *a, **k: [])
    monkeypatch.setattr('hypertrader.bot.fetch_dxy', lambda *a, **k: pd.Series([100]*60))
    monkeypatch.setattr('hypertrader.bot.fetch_interest_rate', lambda *a, **k: pd.Series([5]*60))
    monkeypatch.setattr('hypertrader.bot.fetch_global_liquidity', lambda *a, **k: pd.Series([10]*60))
    signal_file = tmp_path / 'signal.json'
    run('BTC-USD', account_balance=10000, risk_percent=2, news_api_key=None, fred_api_key='key', model_path=None, signal_path=str(signal_file))

    assert signal_file.exists()
    data = json.loads(signal_file.read_text())
    assert 'action' in data

