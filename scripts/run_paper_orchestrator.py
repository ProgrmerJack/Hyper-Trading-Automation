#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
import yaml

# Ensure project root is on sys.path when running from scripts/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hypertrader.orchestrator import TradingOrchestrator  # noqa: E402


def main() -> None:
    # Ensure data directory exists for state files
    (ROOT / "data").mkdir(exist_ok=True)
    cfg_path = ROOT / "config.sample.yaml"
    raw = yaml.safe_load(cfg_path.read_text()) or {}
    trading = raw.get("trading", {})
    symbol = (trading.get("symbols") or ["BTC/USDT"])[0].replace("/", "-")
    cfg = {
        "symbol": symbol,
        "account_balance": trading.get("account_balance", 100.0),
        "risk_percent": trading.get("risk_percent", 1.0),
        "signal_path": "data/signal.json",
        "state_path": "data/state.json",
        "exchange": trading.get("exchange"),
        "etherscan_api_key": None,
        "max_exposure": trading.get("max_exposure", 1.0),
        "live": False,
    }
    TradingOrchestrator(cfg, loop_interval=60.0, use_websocket=True).start()


if __name__ == "__main__":
    main()


