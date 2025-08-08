from __future__ import annotations

import math
from typing import Dict, Any


def validate_order(price: float, quantity: float, market: Dict[str, Any]) -> bool:
    """Return ``True`` if the order satisfies basic venue limits.

    Parameters
    ----------
    price:
        Order price. Used to compute notional for cost limits.
    quantity:
        Order quantity in base currency.
    market:
        Market metadata from CCXT, expected to contain ``limits`` with
        ``amount`` and ``cost`` entries.
    """

    limits = market.get("limits", {})
    amount = limits.get("amount", {})
    cost = limits.get("cost", {})

    min_amount = amount.get("min")
    step = amount.get("step")
    min_cost = cost.get("min")

    if min_amount is not None and quantity < min_amount:
        return False

    if step is not None:
        ratio = quantity / step
        if not math.isclose(ratio, round(ratio), rel_tol=0, abs_tol=1e-8):
            return False

    if min_cost is not None and price * quantity < min_cost:
        return False

    return True
