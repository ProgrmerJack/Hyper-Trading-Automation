from hypertrader.utils.net import fetch_with_retry


def test_fetch_with_retry(monkeypatch):
    calls = {"n": 0}

    def func():
        calls["n"] += 1
        if calls["n"] < 2:
            raise ValueError("fail")
        return 42

    result = fetch_with_retry(func, retries=3, delay=0)
    assert result == 42
    assert calls["n"] == 2
