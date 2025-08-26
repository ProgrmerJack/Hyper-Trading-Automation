"""
Anomaly detection functions for market regime analysis.

This module re‑exports the entropy computation and regime
classification utilities from :mod:`hypertrader.utils.anomaly`.  It
exists to simplify imports within the ``hypertrader_plus`` package and
to document the rationale for using entropy as a measure of
market randomness.  See the original implementation for more
information and references.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Any


def compute_entropy(sequence: Iterable[Any]) -> float:
    """Return the Shannon entropy of ``sequence``.

    The result is normalised to the ``[0, 1]`` range where ``0`` denotes
    a perfectly deterministic sequence and ``1`` indicates a uniform
    distribution across observed symbols.
    """

    items = list(sequence)
    if not items:
        return 0.0
    counts = Counter(items)
    total = len(items)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log(p, 2)
    # Normalise by maximum entropy (log of number of unique symbols)
    max_entropy = math.log(len(counts), 2) if len(counts) > 1 else 1.0
    return float(entropy / max_entropy)


def detect_entropy_regime(
    sequence: Iterable[Any],
    low_threshold: float = 0.3,
    high_threshold: float = 0.7,
) -> str:
    """Classify the entropy regime of ``sequence``.

    Parameters
    ----------
    sequence:
        Observed discrete values such as price change directions.
    low_threshold:
        Entropy below this value is considered ``"trending"``.
    high_threshold:
        Entropy above this value is labelled ``"chaotic"``.
    """

    entropy = compute_entropy(sequence)
    if entropy < low_threshold:
        return "trending"
    if entropy > high_threshold:
        return "chaotic"
    return "normal"


__all__ = ["compute_entropy", "detect_entropy_regime"]
