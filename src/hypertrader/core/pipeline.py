from __future__ import annotations

from typing import Protocol, AsyncIterator, Dict, Any


class DataFeed(Protocol):
    """Protocol for async market data feeds."""

    def stream(self) -> AsyncIterator[Dict[str, Any]]:
        ...


class Strategy(Protocol):
    """Protocol for strategy components producing trade signals."""

    def on_tick(self, tick: Dict[str, Any]) -> Dict[str, Any] | None:
        ...


class RiskGate(Protocol):
    """Protocol for pre-trade risk checks."""

    def check_order(self, equity: float, position_value: float, edge: float) -> bool:
        ...


class OrderExecutor(Protocol):
    """Protocol for order execution components."""

    async def execute(self, signal: Dict[str, Any]) -> None:
        ...


class Pipeline:
    """Event-driven trading pipeline combining feed, strategy, risk and execution."""

    def __init__(
        self,
        feed: DataFeed,
        strategy: Strategy,
        risk: RiskGate,
        executor: OrderExecutor,
    ) -> None:
        self.feed = feed
        self.strategy = strategy
        self.risk = risk
        self.executor = executor

    async def run(self, equity: float) -> None:
        """Continuously process ticks from the data feed."""
        async for tick in self.feed.stream():
            signal = self.strategy.on_tick(tick)
            if not signal:
                continue
            notional = float(signal.get("notional", 0))
            edge = float(signal.get("edge", 0))
            if not self.risk.check_order(equity, notional, edge):
                continue
            await self.executor.execute(signal)
