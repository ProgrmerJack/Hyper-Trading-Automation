import pytest

from hypertrader.orchestrator import TradingOrchestrator


@pytest.mark.asyncio
async def test_orchestrator_runs_once(monkeypatch):
    calls = {"count": 0}

    async def fake_run(**kwargs):
        calls["count"] += 1

    monkeypatch.setattr("hypertrader.orchestrator._run", fake_run)

    orch = TradingOrchestrator({"symbol": "BTC-USD"}, loop_interval=0.0, max_cycles=1, use_websocket=False)
    await orch.run_loop()

    assert calls["count"] == 1
