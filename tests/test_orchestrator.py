from hypertrader.orchestrator import TradingOrchestrator


def test_orchestrator_runs_once(monkeypatch):
    calls = {
        "count": 0,
    }

    def fake_run(**kwargs):
        calls["count"] += 1

    monkeypatch.setattr("hypertrader.orchestrator.run", fake_run)

    orch = TradingOrchestrator({"symbol": "BTC-USD"}, loop_interval=0.0, max_cycles=1)
    orch.run_loop()

    assert calls["count"] == 1
