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

from hypertrader.utils.anomaly import compute_entropy, detect_entropy_regime


__all__ = ["compute_entropy", "detect_entropy_regime"]
