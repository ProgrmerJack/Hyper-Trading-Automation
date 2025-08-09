import pytest

from hypertrader.execution import ccxt_executor


@pytest.mark.asyncio
async def test_post_only_not_supported(monkeypatch):
    class DummyExchange:
        has = {"createPostOnlyOrder": False, "createReduceOnlyOrder": True}

        async def load_markets(self):
            pass

        def market(self, symbol):
            return {"limits": {"amount": {"min": 0}}}

        async def create_limit_buy_order(self, symbol, qty, price, params):
            return {}

    monkeypatch.setattr(ccxt_executor, "ex", DummyExchange())
    monkeypatch.setattr(ccxt_executor, "validate_order", lambda p, q, m: True)

    with pytest.raises(RuntimeError):
        await ccxt_executor.place_order("BTC/USDT", "buy", 1, 10, post_only=True)


@pytest.mark.asyncio
async def test_flags_applied(monkeypatch):
    captured = {}

    class DummyExchange:
        has = {"createPostOnlyOrder": True, "createReduceOnlyOrder": True}

        async def load_markets(self):
            pass

        def market(self, symbol):
            return {"limits": {"amount": {"min": 0}}}

        async def create_limit_sell_order(self, symbol, qty, price, params):
            captured["params"] = params
            return {}

    monkeypatch.setattr(ccxt_executor, "ex", DummyExchange())
    monkeypatch.setattr(ccxt_executor, "validate_order", lambda p, q, m: True)

    await ccxt_executor.place_order("BTC/USDT", "sell", 1, 10, post_only=True, reduce_only=True)
    assert captured["params"]["postOnly"] is True
    assert captured["params"]["reduceOnly"] is True
