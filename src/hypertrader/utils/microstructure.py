"""Lightweight market microstructure utilities.

The original project exposes a richer set of functions for analysing
order book dynamics.  For the purposes of the tests we only need a
small subset of that functionality, so this module implements minimal
versions of the commonly used helpers.  The implementations favour
clarity over microstructure realism but capture the intended behaviour
for unit tests and example strategies.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple, Any


def compute_microprice(order_book: Dict[str, List[Tuple[float, float]]], depth: int = 1) -> float:
    """Return the micro‑price using the top ``depth`` levels.

    The micro‑price tilts the mid price towards the side with greater
    volume.  When one side of the book is empty the best price from the
    other side is returned.
    """

    bids = order_book.get("bids", [])[:depth]
    asks = order_book.get("asks", [])[:depth]
    if not bids and not asks:
        return 0.0
    best_bid = bids[0][0] if bids else asks[0][0]
    best_ask = asks[0][0] if asks else bids[0][0]
    bid_vol = sum(v for _, v in bids) or 1.0
    ask_vol = sum(v for _, v in asks) or 1.0
    return (best_bid * ask_vol + best_ask * bid_vol) / (bid_vol + ask_vol)


def flow_toxicity(trades: Iterable[dict], window: int = 100) -> float:
    """Return a simple order flow imbalance metric.

    The function looks at the most recent ``window`` trades and computes
    the absolute difference between buy and sell volume normalised by
    total volume.
    """

    recent = list(trades)[-window:]
    buy_vol = sum(t.get("amount", 0.0) for t in recent if t.get("side") == "buy")
    sell_vol = sum(t.get("amount", 0.0) for t in recent if t.get("side") == "sell")
    total = buy_vol + sell_vol
    if total == 0:
        return 0.0
    return abs(buy_vol - sell_vol) / total


def detect_iceberg(
    order_book: Dict[str, List[Tuple[float, float]]],
    traded_volume_at_top: float,
    factor: float = 2.0,
) -> bool:
    """Heuristically flag potential iceberg orders.

    If the traded volume at the best price exceeds ``factor`` times the
    displayed volume, an iceberg order (hidden liquidity) is suspected.
    """

    bids = order_book.get("bids", [])
    asks = order_book.get("asks", [])
    top_bid = bids[0][1] if bids else 0.0
    top_ask = asks[0][1] if asks else 0.0
    top_vol = max(top_bid, top_ask)
    return traded_volume_at_top > factor * top_vol


def compute_entropy(sequence: Iterable[Any]) -> float:
    """Proxy to :func:`hypertrader.utils.anomaly.compute_entropy`."""

    from .anomaly import compute_entropy as _compute_entropy

    return _compute_entropy(sequence)


__all__ = [
    "compute_microprice",
    "flow_toxicity",
    "detect_iceberg",
    "compute_entropy",
]

