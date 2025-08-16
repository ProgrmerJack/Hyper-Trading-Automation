"""
Microstructure indicators for high‑frequency trading.

This module collects functions that extract information from limit
order books and order flow.  They serve as lightweight proxies for
more sophisticated market microstructure models.  The functions
provided include:

* ``compute_microprice`` – compute the micro‑price using the best
  bid/ask and their volumes.  It tilts the mid price towards the
  side with greater resting liquidity and acts as a short‑horizon
  estimator of future price movements.
* ``flow_toxicity`` – return a measure of order flow imbalance
  analogous to the VPIN.  It reports values close to 1 when recent
  trades are heavily skewed in one direction and 0 for balanced flow.
* ``compute_entropy`` – compute the Shannon entropy of a discrete
  sequence.  This can be used to quantify randomness in price
  changes or order book states.
* ``detect_entropy_regime`` – classify the entropy of a sequence as
  ``"trending"``, ``"normal"`` or ``"chaotic"`` based on thresholds.
* ``detect_iceberg`` – heuristically flag iceberg orders by
  comparing traded volume at the best price to the displayed depth.

These functions are wrappers around implementations in
``hypertrader.utils.microstructure`` and ``hypertrader.utils.anomaly``.
They are exposed here to keep external dependencies contained and to
allow experimentation with alternative metrics without modifying the
core utilities.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple, Any

from hypertrader.utils.microstructure import (
    compute_microprice as _compute_microprice,
    flow_toxicity as _flow_toxicity,
    compute_entropy as _compute_entropy,
    detect_iceberg as _detect_iceberg,
)
from hypertrader.utils.anomaly import detect_entropy_regime as _detect_entropy_regime


def compute_microprice(order_book: Dict[str, List[Tuple[float, float]]], depth: int = 1) -> float:
    """Compute the micro‑price using the implementation from
    :mod:`hypertrader.utils.microstructure`.

    This wrapper exists to simplify imports and documentation in the
    ``hypertrader_plus`` namespace.  See the original function for
    parameter and return descriptions.
    """
    return _compute_microprice(order_book, depth)


def flow_toxicity(trades: Iterable[dict], window: int = 100) -> float:
    """Return a volume imbalance metric approximating VPIN.

    Delegates to :func:`hypertrader.utils.microstructure.flow_toxicity`.
    """
    # Convert iterable to list to support slicing in underlying impl
    return _flow_toxicity(list(trades), window)


def compute_entropy(sequence: Iterable[Any]) -> float:
    """Compute the Shannon entropy of ``sequence``.

    Simply calls :func:`hypertrader.utils.microstructure.compute_entropy`.
    """
    return _compute_entropy(list(sequence))


def detect_entropy_regime(sequence: Iterable[Any], low_threshold: float = 0.3, high_threshold: float = 0.7) -> str:
    """Classify the entropy regime of a sequence.

    Wraps :func:`hypertrader.utils.anomaly.detect_entropy_regime`.  See
    that function for parameter documentation.
    """
    return _detect_entropy_regime(list(sequence), low_threshold, high_threshold)


def detect_iceberg(order_book: Dict[str, List[Tuple[float, float]]], traded_volume_at_top: float, factor: float = 2.0) -> bool:
    """Detect potential iceberg orders by comparing executed volume at the top of book to displayed size.

    Calls :func:`hypertrader.utils.microstructure.detect_iceberg`.
    """
    return _detect_iceberg(order_book, traded_volume_at_top, factor)


__all__ = [
    "compute_microprice",
    "flow_toxicity",
    "compute_entropy",
    "detect_entropy_regime",
    "detect_iceberg",
]
