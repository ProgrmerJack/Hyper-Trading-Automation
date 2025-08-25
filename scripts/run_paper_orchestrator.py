#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import os
import sys

# Ensure project root is on sys.path when running from scripts/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hypertrader.orchestrator import TradingOrchestrator


def main() -> None:
    # Ensure data directory exists for state files
    (ROOT / "data").mkdir(exist_ok=True)
    # Force demo activity for visibility
    os.environ.setdefault("DEMO_MODE", "true")
    cfg = {
        "symbol": "BTC-USD",
        "account_balance": 100.0,  # Start with $100 for 10x challenge
        "risk_percent": 5.0,      # Aggressive 5% risk per trade for growth
        "signal_path": "data/signal.json",
        "state_path": "data/state.json",
        "exchange": "binance",
        "etherscan_api_key": None,
        "max_exposure": 10.0,     # Allow higher exposure for growth
        "live": False,
    }
    TradingOrchestrator(cfg, loop_interval=60.0, use_websocket=True).start()


if __name__ == "__main__":
    main()


