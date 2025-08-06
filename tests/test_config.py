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
