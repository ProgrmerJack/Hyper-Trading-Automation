from __future__ import annotations

import math
from typing import Dict, Any


def _is_multiple(value: float, step: float) -> bool:
    if step is None or step == 0:
        return True
    ratio = value / step
    return math.isclose(ratio, round(ratio), rel_tol=0, abs_tol=1e-8)


def validate_order(price: float, quantity: float, market: Dict[str, Any]) -> bool:
    """Return ``True`` if the order satisfies basic venue limits.

    Parameters
    ----------
    price:
        Order price. Used to compute notional for cost limits. When ``<= 0`` it
        is treated as a market order price placeholder and cost checks are
        skipped.
    quantity:
        Order quantity in base currency.
    market:
        Market metadata from CCXT, expected to contain ``limits`` with
        ``amount``/``price`` and ``cost`` entries and optionally ``precision``.
    """

    limits = market.get("limits", {})
    amount_limits = limits.get("amount", {})
    price_limits = limits.get("price", {})
    cost_limits = limits.get("cost", {})
    precision = market.get("precision", {})

    min_amount = amount_limits.get("min")
    amount_step = amount_limits.get("step")
    min_cost = cost_limits.get("min")
    price_step = price_limits.get("step")
    price_min = price_limits.get("min")
    price_precision = precision.get("price")

    # Amount checks -----------------------------------------------------
    if min_amount is not None and quantity < min_amount:
        return False

    if amount_step is not None and not _is_multiple(quantity, amount_step):
        return False

    # Price checks (only for limit-style orders where price > 0) --------
    if price is not None and price > 0:
        if price_min is not None and price < price_min:
            return False
        if price_step is not None:
            if not _is_multiple(price, price_step):
                return False
        elif isinstance(price_precision, int):
            # Ensure price conforms to precision if step is not provided
            rounded = round(price, price_precision)
            if not math.isclose(price, rounded, rel_tol=0, abs_tol=1e-8):
                return False

        if min_cost is not None and price * quantity < min_cost:
            return False

    # For market orders (price <= 0), skip min_cost check because notional
    # will be determined by the venue at execution time.
    return True
