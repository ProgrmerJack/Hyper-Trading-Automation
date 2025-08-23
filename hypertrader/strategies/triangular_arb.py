"""
Triangular arbitrage strategy.

This strategy monitors three currency pairs that form a closed loop
and exploits pricing inconsistencies among them.  In a triangular
loop such as ``A/B``, ``B/C``, and ``A/C``, the product of the
exchange rates should equal one under no‑arbitrage conditions (up to
fees).  If the product deviates significantly from unity, there is an
opportunity to trade around the loop and realise a risk‑free profit
in theory.  In practice, latency and execution costs reduce or
eliminate profits, but the strategy remains useful for studying
market efficiency.

This implementation is simplified: it takes pre‑computed mid prices
for the three pairs, computes the loop product, and generates a set
of market orders when the product exceeds a threshold.  The
quantities are equal across legs and not adjusted for volume or
order book depth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Union


@dataclass
class TriangularArbitrageStrategy:
    """Simple triangular arbitrage based on price multiplication.

    Parameters
    ----------
    pair_ab : str
        Symbol for the first leg (e.g., ``"BTC/USDT"``).
    pair_bc : str
        Symbol for the second leg (e.g., ``"USDT/ETH"``).  Note that
        the quote currency of ``pair_ab`` should match the base
        currency of ``pair_bc``.
    pair_ac : str
        Symbol for the third leg (e.g., ``"BTC/ETH"``).
    threshold : float, optional
        Minimum deviation from 1.0 required to trigger a trade.  For
        example, a value of 0.002 means a 0.2% discrepancy.  Default
        is 0.001.
    size : float, optional
        Nominal size to trade on each leg.  Default is 1.0.  The
        actual notional amounts depend on the prices of the legs.
    """

    pair_ab: str
    pair_bc: str
    pair_ac: str
    threshold: float = 0.001
    size: float = 1.0

    def update(self, price_ab: float, price_bc: float, price_ac: float) -> List[Tuple[str, str, float]]:
        """Detect arbitrage and return the trades to execute.

        Parameters
        ----------
        price_ab : float
            Mid or microprice for pair A/B.
        price_bc : float
            Mid or microprice for pair B/C.
        price_ac : float
            Mid or microprice for pair A/C.

        Returns
        -------
        list of tuple
            A list of market orders ``(side, symbol, quantity)`` to
            execute in order around the loop.  The quantity
            corresponds to ``size`` expressed in the base currency of
            the first leg.  An empty list indicates no arbitrage.
        """
        # Compute the implied price of A/C via AB and BC: A/B * B/C
        implied_ac = price_ab * price_bc
        diff = implied_ac - price_ac
        relative = diff / price_ac if price_ac != 0 else 0.0
        orders: List[Tuple[str, str, float]] = []
        if relative > self.threshold:
            # implied > actual: buy A/B and B/C, sell A/C
            # Equivalent: convert A to B to C and back to A for profit
            orders.append(("buy", self.pair_ab, self.size))
            orders.append(("buy", self.pair_bc, self.size))
            orders.append(("sell", self.pair_ac, self.size))
        elif relative < -self.threshold:
            # implied < actual: reverse cycle
            orders.append(("buy", self.pair_ac, self.size))
            orders.append(("sell", self.pair_bc, self.size))
            orders.append(("sell", self.pair_ab, self.size))
        return orders


@dataclass
class TriangularArb:
    """Minimal triangular arbitrage signal detector."""
    threshold: float = 0.0005
    
    def signal(self, p_ab: float, p_bc: float, p_ac: float) -> str | None:
        """Detect triangular arbitrage opportunity."""
        loop = p_ab * p_bc / p_ac
        if loop > 1 + self.threshold:
            return 'A->B->C->A'
        if loop < 1 - self.threshold:
            return 'A->C->B->A'
        return None


def triangular_signal(p_ab: float, p_bc: float, p_ac: float, 
                     threshold: float = 0.0005) -> Tuple[str | None, float]:
    """Generate triangular arbitrage signal and loop ratio."""
    loop = p_ab * p_bc / p_ac
    signal = None
    if loop > 1 + threshold:
        signal = 'A->B->C->A'
    elif loop < 1 - threshold:
        signal = 'A->C->B->A'
    return signal, loop
