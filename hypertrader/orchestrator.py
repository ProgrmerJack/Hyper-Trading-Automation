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
from .execution.ccxt_executor import cancel_all, ex


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
            api_key = getattr(ex, "apiKey", None)
            api_secret = getattr(ex, "secret", None)
            user_feed = PrivateWebSocketFeed(exchange, store, api_key, api_secret)
            user_task = asyncio.create_task(user_feed.run())

        try:
            if self.use_websocket and isinstance(symbol, str) and exchange:
                ws_symbol = symbol
                ccxt_symbol = symbol.replace("-", "/") if isinstance(symbol, str) else symbol
                if exchange.lower() == "binance":
                    ws_symbol = symbol.replace("-", "").replace("/", "").lower()
                elif exchange.lower() == "bybit":
                    ws_symbol = symbol.replace("/", "").replace("-", "").upper()
                feed = ExchangeWebSocketFeed(exchange, ws_symbol)
                candle_queue: asyncio.Queue[list[float]] = asyncio.Queue()
                candle_task = asyncio.create_task(
                    stream_ohlcv(ccxt_symbol, exchange_name=exchange, queue=candle_queue)
                )
                candles: list[list[float]] = []
                try:
                    async for msg in feed.stream():
                        if msg is None:
                            # heartbeat missed -> cancel outstanding orders
                            try:
                                await cancel_all()
                            except Exception as e:
                                import logging
                                logging.warning(f"Order cancellation failed: {e}")
                                pass
                            continue
                        try:
                            candle = candle_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            continue
                        candles.append(candle)
                        candles = candles[-1000:]
                        df = pd.DataFrame(
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
                    candle_task.cancel()
                    with contextlib.suppress(Exception):
                        await candle_task
                    await feed.close()
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
