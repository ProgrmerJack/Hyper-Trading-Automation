from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RiskParams:
    """Configuration controlling pre-trade risk checks."""

    max_daily_loss: float
    """Absolute loss in account currency that triggers trading halt."""

    max_position: float
    """Maximum allowed notional exposure per order."""

    fee_rate: float = 0.0
    """Estimated taker fee rate used to sanity-check signal edge."""


class RiskManager:
    """Simple risk gate evaluated before every order.

    The manager keeps track of starting equity for the day and ensures
    that new orders respect exposure limits and halt trading once the
    daily loss limit has been reached.
    """

    def __init__(self, params: RiskParams) -> None:
        self.params = params
        self.starting_equity: float | None = None

    def reset_day(self, equity: float) -> None:
        """Reset the starting equity, typically at the beginning of a session."""
        self.starting_equity = equity

    def check_order(self, equity: float, position_value: float, edge: float) -> bool:
        """Return ``True`` if the order passes all risk checks."""
        if self.starting_equity is None:
            self.reset_day(equity)
        loss = self.starting_equity - equity
        if loss > self.params.max_daily_loss:
            return False
        if position_value > self.params.max_position:
            return False
        if edge <= self.params.fee_rate:
            return False
        return True
