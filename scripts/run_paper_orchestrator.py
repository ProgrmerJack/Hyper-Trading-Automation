#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

# Ensure project root is on sys.path when running from scripts/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hypertrader.orchestrator import TradingOrchestrator


def main() -> None:
    # Ensure data directory exists for state files
    (ROOT / "data").mkdir(exist_ok=True)
    cfg = {
        "symbol": "BTC-USD",
        "account_balance": 10000.0,
        "risk_percent": 1.0,
        "signal_path": "data/signal.json",
        "state_path": "data/state.json",
        "exchange": "binance",
        "etherscan_api_key": None,
        "max_exposure": 3.0,
        "live": False,
    }
    TradingOrchestrator(cfg, loop_interval=5.0, use_websocket=True).start()


if __name__ == "__main__":
    main()


