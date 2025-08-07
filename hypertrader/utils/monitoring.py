"""Monitoring and anomaly detection utilities."""
from __future__ import annotations

from typing import Iterable

import numpy as np
from prometheus_client import Gauge, start_http_server
from sklearn.ensemble import IsolationForest

# Prometheus metric for tracking trade latency
latency_gauge = Gauge("trade_latency_ms", "Execution latency in milliseconds")


def start_metrics_server(port: int = 8000) -> None:
    """Start a Prometheus metrics HTTP server."""
    start_http_server(port)


def monitor_latency(value: float) -> None:
    """Update the latency gauge."""
    latency_gauge.set(value)


def detect_anomalies(metrics: Iterable[float]) -> np.ndarray:
    """Detect anomalies in a sequence of metrics using Isolation Forest.

    Parameters
    ----------
    metrics : Iterable[float]
        Sequence of metric values such as latencies or PnL.

    Returns
    -------
    numpy.ndarray
        Array of labels ``1`` for normal observations and ``-1`` for anomalies.
    """
    arr = np.fromiter(metrics, dtype=float)
    model = IsolationForest(contamination="auto")
    return model.fit_predict(arr.reshape(-1, 1))


__all__ = ["start_metrics_server", "monitor_latency", "detect_anomalies", "latency_gauge"]
