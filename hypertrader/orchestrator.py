"""Central orchestration for the trading system."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import time

from .bot import run


@dataclass
class TradingOrchestrator:
    """High level orchestrator coordinating the trading pipeline.

    Parameters
    ----------
    config : dict
        Configuration passed directly to :func:`hypertrader.bot.run`.
    loop_interval : float, default ``60.0``
        Seconds to sleep between iterations.
    max_cycles : int, optional
        If provided, the loop stops after this many iterations.
    """

    config: Dict[str, Any]
    loop_interval: float = 60.0
    max_cycles: Optional[int] = None

    def run_loop(self) -> None:
        """Execute the trading loop."""
        cycles = 0
        while self.max_cycles is None or cycles < self.max_cycles:
            run(**self.config)
            cycles += 1
            time.sleep(self.loop_interval)
