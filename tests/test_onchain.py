import pandas as pd

from hypertrader.data.onchain import fetch_eth_gas_fees


def test_fetch_eth_gas_fees(monkeypatch):
    class DummyResponse:
        def __init__(self):
            self.status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"result": {"ProposeGasPrice": "12.3"}}

    def dummy_get(url, params=None, timeout=10):
        return DummyResponse()

    monkeypatch.setattr("requests.get", dummy_get)
    df = fetch_eth_gas_fees(api_key="dummy")
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["gas"]
    assert df.iloc[0, 0] == 12.3
