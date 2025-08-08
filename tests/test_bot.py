import json
import pandas as pd

from hypertrader.bot import run


def test_run_creates_signal(monkeypatch, tmp_path):
    def dummy_download(symbol, period='7d', interval='1h', progress=False):
        idx = pd.date_range('2024-01-01', periods=200, freq='h')
        data = {
            'open': [1]*200,
            'high': [1]*200,
            'low': [1]*200,
            'close': [1]*200,
            'volume': [1]*200,
        }
        return pd.DataFrame(data, index=idx)

    monkeypatch.setattr('hypertrader.bot.fetch_ohlcv', lambda *a, **k: dummy_download(None))
    monkeypatch.setattr('hypertrader.bot.fetch_news_headlines', lambda *a, **k: [])
    monkeypatch.setattr('hypertrader.bot.fetch_dxy', lambda *a, **k: pd.Series([100]*60))
    monkeypatch.setattr('hypertrader.bot.fetch_interest_rate', lambda *a, **k: pd.Series([5]*60))
    monkeypatch.setattr('hypertrader.bot.fetch_global_liquidity', lambda *a, **k: pd.Series([10]*60))
    signal_file = tmp_path / 'signal.json'
    run('BTC-USD', account_balance=10000, risk_percent=2, news_api_key=None, fred_api_key='key', model_path=None, signal_path=str(signal_file))

    assert signal_file.exists()
    data = json.loads(signal_file.read_text())
    assert 'action' in data
    state_file = signal_file.with_name('state.json')
    assert state_file.exists()
    state = json.loads(state_file.read_text())
    assert state['equity'] == 10000
    assert len(state['latencies']) == 1


def test_kill_switch_halts_trading(monkeypatch, tmp_path):
    def dummy_download(symbol, period='7d', interval='1h', progress=False):
        idx = pd.date_range('2024-01-01', periods=200, freq='h')
        data = {
            'open': [1]*200,
            'high': [1]*200,
            'low': [1]*200,
            'close': [1]*200,
            'volume': [1]*200,
        }
        return pd.DataFrame(data, index=idx)

    monkeypatch.setattr('hypertrader.bot.fetch_ohlcv', lambda *a, **k: dummy_download(None))
    monkeypatch.setattr('hypertrader.bot.fetch_news_headlines', lambda *a, **k: [])
    monkeypatch.setattr('hypertrader.bot.fetch_dxy', lambda *a, **k: pd.Series([100]*60))
    monkeypatch.setattr('hypertrader.bot.fetch_interest_rate', lambda *a, **k: pd.Series([5]*60))
    monkeypatch.setattr('hypertrader.bot.fetch_global_liquidity', lambda *a, **k: pd.Series([10]*60))

    signal_file = tmp_path / 'signal.json'
    state_file = tmp_path / 'state.json'
    state_file.write_text(json.dumps({'peak_equity': 10000}))

    run('BTC-USD', account_balance=8000, risk_percent=2, news_api_key=None,
        fred_api_key='key', model_path=None, signal_path=str(signal_file),
        state_path=str(state_file))

    data = json.loads(signal_file.read_text())
    assert data['action'] == 'HOLD'
    state = json.loads(state_file.read_text())
    assert len(state['latencies']) == 1


def test_run_uses_onchain_and_orderbook(monkeypatch, tmp_path):
    import numpy as np

    def dummy_download(symbol, period='7d', interval='1h', progress=False):
        idx = pd.date_range('2024-01-01', periods=200, freq='h')
        # create gentle uptrend to satisfy strategy conditions
        closes = np.linspace(1, 1.2, 200)
        data = {
            'open': closes,
            'high': closes + 0.1,
            'low': closes - 0.1,
            'close': closes,
            'volume': [1]*200,
        }
        return pd.DataFrame(data, index=idx)

    gas_df = pd.DataFrame({'gas': list(range(1, 31))})
    gas_df.index = pd.date_range('2024-01-01', periods=30, freq='h')

    order_book = {"bids": [[1, 10]], "asks": [[1, 1]]}

    from hypertrader.strategies.indicator_signals import Signal

    def dummy_generate(data, sentiment, macro, onchain, skew, heatmap):
        assert onchain > 1.0
        assert skew > 0.2
        assert heatmap > 1.2
        return Signal('BUY')

    monkeypatch.setattr('hypertrader.bot.fetch_ohlcv', lambda *a, **k: dummy_download(None))
    monkeypatch.setattr('hypertrader.bot.fetch_news_headlines', lambda *a, **k: [])
    monkeypatch.setattr('hypertrader.bot.fetch_dxy', lambda *a, **k: pd.Series([100]*60))
    monkeypatch.setattr('hypertrader.bot.fetch_interest_rate', lambda *a, **k: pd.Series([5]*60))
    monkeypatch.setattr('hypertrader.bot.fetch_global_liquidity', lambda *a, **k: pd.Series([10]*60))
    monkeypatch.setattr('hypertrader.bot.fetch_eth_gas_fees', lambda *a, **k: gas_df)
    monkeypatch.setattr('hypertrader.bot.fetch_order_book', lambda *a, **k: order_book)
    monkeypatch.setattr('hypertrader.bot.generate_signal', dummy_generate)
    monkeypatch.setattr('hypertrader.bot.start_metrics_server', lambda *a, **k: None)
    monkeypatch.setattr('hypertrader.bot.drl_throttle', lambda state: 1.0)

    signal_file = tmp_path / 'signal.json'
    run(
        'BTC-USD',
        account_balance=10000,
        risk_percent=2,
        news_api_key=None,
        fred_api_key='key',
        model_path=None,
        signal_path=str(signal_file),
        exchange='binance',
        etherscan_api_key='dummy',
    )

    data = json.loads(signal_file.read_text())
    assert data['action'] == 'BUY'

