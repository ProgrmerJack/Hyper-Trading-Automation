"""Central orchestration for the trading system."""
from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from .bot import _run
from .feeds.exchange_ws import ExchangeWebSocketFeed
from .feeds.private_ws import PrivateWebSocketFeed
from .data.fetch_data import stream_ohlcv
from .data.oms_store import OMSStore


@dataclass
class TradingOrchestrator:
    """High level orchestrator coordinating the trading pipeline.

    Parameters
    ----------
    config : dict
        Configuration passed directly to :func:`hypertrader.bot._run`.
    loop_interval : float, default ``60.0``
        Seconds to sleep between iterations when not streaming.
    max_cycles : int, optional
        If provided, the loop stops after this many iterations.
    use_websocket : bool, default ``True``
        When ``True`` and both ``symbol`` and ``exchange`` are supplied in
        ``config`` the orchestrator subscribes to a WebSocket ticker feed and
        triggers a trading cycle on each update.  Falls back to a timed loop
        otherwise.
    """

    config: Dict[str, Any]
    loop_interval: float = 60.0
    max_cycles: Optional[int] = None
    use_websocket: bool = True

    async def _cycle(self, data: pd.DataFrame | None = None) -> None:
        """Run a single trading cycle."""
        await _run(data=data, **self.config)

    async def run_loop(self) -> None:
        """Execute the trading loop using WebSocket events or timed sleeps."""
        cycles = 0
        symbol = self.config.get("symbol")
        exchange = self.config.get("exchange")

        state_path = self.config.get("state_path")
        signal_path = self.config.get("signal_path", "signal.json")
        db_path = Path(state_path or signal_path).with_suffix(".db")
        store = OMSStore(db_path)
        self.config["store"] = store
        user_task = None
        if self.config.get("live") and exchange:
            try:
                from .execution.ccxt_executor import ex as _ex
                api_key = str(getattr(_ex, "apiKey", "") or "")
                api_secret = str(getattr(_ex, "secret", "") or "")
            except Exception:
                api_key = ""
                api_secret = ""
            user_feed = PrivateWebSocketFeed(exchange, store, api_key, api_secret)
            user_task = asyncio.create_task(user_feed.run())

        try:
            if self.use_websocket and isinstance(symbol, str) and exchange:
                # Normalize symbols per venue
                if exchange.lower() == "binance":
                    base, _, quote = symbol.replace("/", "-").partition("-")
                    quote = "USDT"  # map USD-like to USDT on Binance
                    ccxt_symbol = f"{base}/{quote}"
                    ws_symbol = f"{base}{quote}".lower()
                elif exchange.lower() == "bybit":
                    base, _, quote = symbol.replace("/", "-").partition("-")
                    quote = quote or "USDT"
                    ccxt_symbol = f"{base}/{quote}"
                    ws_symbol = f"{base}{quote}".upper()
                else:
                    ccxt_symbol = symbol.replace("-", "/")
                    ws_symbol = symbol.replace("-", "").replace("/", "")
                
                # Use stream_ohlcv directly - it manages its own WebSocket feed
                candles: list[list[float]] = []
                try:
                    async for candle in stream_ohlcv(
                        ccxt_symbol, exchange_name=exchange
                    ):
                        if candle is None:
                            # heartbeat missed -> cancel outstanding orders
                            try:
                                from .execution.ccxt_executor import cancel_all as _cancel_all
                                await _cancel_all()
                            except Exception as e:
                                import logging
                                logging.warning(f"Order cancellation failed: {e}")
                                pass
                            continue
                        candles.append(candle)
                        candles = candles[-1000:]
                        df = pd.DataFrame.from_records(
                            candles,
                            columns=["timestamp", "open", "high", "low", "close", "volume"],
                        )
                        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                        df.set_index("timestamp", inplace=True)
                        await self._cycle(df)
                        cycles += 1
                        if self.max_cycles is not None and cycles >= self.max_cycles:
                            break
                finally:
                    pass  # stream_ohlcv manages its own cleanup
            else:
                while self.max_cycles is None or cycles < self.max_cycles:
                    await self._cycle()
                    cycles += 1
                    await asyncio.sleep(self.loop_interval)
        finally:
            if user_task:
                user_task.cancel()
                with contextlib.suppress(Exception):
                    await user_task
            await store.close()

    def start(self) -> None:
        """Blocking entry point that starts the asyncio loop."""
        asyncio.run(self.run_loop())
