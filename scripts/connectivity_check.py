#!/usr/bin/env python3
"""Connectivity and dry-run check for Hyper-Trading Automation.

Runs a short sequence to verify:
- Environment variables are loadable (dotenv)
- CCXT REST connectivity and market metadata
- Public WebSocket reception for ticker
- SQLite OMS store write/read
- Paper signal generation via bot (no live orders)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from __future__ import annotations

import asyncio
import os
import time
import ccxt
from dotenv import load_dotenv

from hypertrader.data.fetch_data import fetch_ohlcv
from hypertrader.feeds.exchange_ws import ExchangeWebSocketFeed
from hypertrader.data.oms_store import OMSStore
from hypertrader.bot import run as bot_run


async def ws_probe(exchange: str, symbol: str, timeout: float = 5.0) -> bool:
    ws_symbol = symbol
    if exchange.lower() == "binance":
        ws_symbol = symbol.replace("/", "").replace("-", "").lower()
    elif exchange.lower() == "bybit":
        ws_symbol = symbol.replace("/", "").replace("-", "").upper()
    feed = ExchangeWebSocketFeed(exchange, ws_symbol)
    try:
        agen = feed.stream()
        # Advance the async generator once with a controlled timeout
        msg = await asyncio.wait_for(agen.__anext__(), timeout=timeout)
        return bool(msg)
    except Exception:
        return False
    finally:
        try:
            await feed.close()
        except Exception:
            pass


def main() -> int:
    print("[1/5] Loading environment (.env)…")
    load_dotenv()
    exchange = os.getenv("EXCHANGE", "binance")
    api_key = os.getenv("API_KEY")
    api_secret = os.getenv("API_SECRET")
    print(f"    EXCHANGE={exchange}")
    if api_key and api_secret:
        print("    API keys: present")
    else:
        print("    API keys: missing (ok for paper mode)")

    symbol = os.getenv("CHECK_SYMBOL", "BTC/USDT")

    print("[2/5] Verifying CCXT REST and markets…")
    try:
        ex = getattr(ccxt, exchange)({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
        })
        ex.load_markets()
        market = ex.market(symbol)
        print(f"    Market loaded: {market.get('symbol', symbol)}")
    except Exception as e:
        print(f"    CCXT error: {e}")
        return 1

    print("[3/5] Fetching sample OHLCV…")
    try:
        df = fetch_ohlcv(exchange, symbol, timeframe="1m", limit=50)
        print(f"    Candles: {len(df)}; last close={df['close'].iloc[-1]:.4f}")
    except Exception as e:
        print(f"    OHLCV error: {e}")
        return 1

    print("[4/5] Probing public WebSocket…")
    ok = asyncio.run(ws_probe(exchange, symbol))
    print(f"    WebSocket: {'OK' if ok else 'FAILED'}")
    if not ok:
        return 1

    print("[5/5] Testing OMS store and paper signal…")
    db_path = Path("data/state.db")
    store = OMSStore(db_path)
    now = time.time()
    asyncio.run(store.record_order("test-1", "cid-1", symbol, "buy", 0.001, 10000.0, "NEW", now))
    open_after = list(asyncio.run(store.fetch_open_orders()))
    print(f"    OMS open orders rows: {len(open_after)}")
    asyncio.run(store.close())

    # Paper signal generation
    try:
        from hypertrader.utils.logging import get_logger, log_json
        print("    Running paper bot one-shot…")
        bot_run(symbol, account_balance=1000.0, risk_percent=1.0, live=False)
        print("    Paper bot run: OK (signal.json written)")
    except Exception as e:
        print(f"    Paper bot error: {e}")
        return 1

    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
