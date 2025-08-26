from __future__ import annotations
from dataclasses import dataclass
@dataclass
class RiskPolicy:
    max_leverage: float = 3.0
    kelly_cap: float = 0.2
    day_loss_limit: float = 0.05
    day_profit_lock: float = 0.03
    dd_limit: float = 0.15
    equity_peak: float = 1000.0
    def kelly_fraction(self, win_rate: float, payoff: float) -> float:
        p = max(0.0, min(1.0, win_rate)); b = max(1e-9, payoff)
        f = p - (1-p)/b
        return max(0.0, min(self.kelly_cap, f))
    def daily_controls(self, day_pnl_frac: float) -> float:
        if day_pnl_frac <= -self.day_loss_limit: return 0.0
        if day_pnl_frac >= self.day_profit_lock: return 0.5
        return 1.0
    def dd_control(self, equity: float) -> float:
        self.equity_peak = max(self.equity_peak, equity)
        dd = (self.equity_peak - equity)/max(1e-9, self.equity_peak)
        return 0.0 if dd >= self.dd_limit else 1.0
