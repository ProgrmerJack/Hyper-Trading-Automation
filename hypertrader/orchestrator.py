"""Central orchestration for the trading system."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .bot import _run
from .feeds.ccxt_ws import CCXTWebSocketFeed


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

    async def _cycle(self) -> None:
        """Run a single trading cycle."""
        await _run(**self.config)

    async def run_loop(self) -> None:
        """Execute the trading loop using WebSocket events or timed sleeps."""
        cycles = 0
        symbol = self.config.get("symbol")
        exchange = self.config.get("exchange")

        if self.use_websocket and isinstance(symbol, str) and exchange:
            feed = CCXTWebSocketFeed(exchange, symbol.replace("-", "/"))
            try:
                async for _ in feed.stream():
                    await self._cycle()
                    cycles += 1
                    if self.max_cycles is not None and cycles >= self.max_cycles:
                        break
            finally:
                await feed.close()
        else:
            while self.max_cycles is None or cycles < self.max_cycles:
                await self._cycle()
                cycles += 1
                await asyncio.sleep(self.loop_interval)

    def start(self) -> None:
        """Blocking entry point that starts the asyncio loop."""
        asyncio.run(self.run_loop())
