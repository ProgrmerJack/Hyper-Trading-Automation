from pathlib import Path

import yaml

from hypertrader.config import load_config


def test_load_config(tmp_path: Path):
    cfg = {
        "api_keys": {"fred": "X"},
        "trading": {"symbol": "BTC-USD", "risk_percent": 1.0},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))

    loaded = load_config(path)
    assert loaded["trading"]["symbol"] == "BTC-USD"
    assert loaded["trading"]["risk_percent"] == 1.0


def test_env_api_keys(monkeypatch, tmp_path: Path):
    cfg = {"trading": {"symbol": "BTC-USD", "risk_percent": 1.0}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv("FRED_API_KEY", "ENVKEY")
    loaded = load_config(path)
    assert loaded["api_keys"]["fred"] == "ENVKEY"
