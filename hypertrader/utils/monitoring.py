"""Monitoring and anomaly detection utilities."""
from __future__ import annotations

from typing import Iterable

import numpy as np
from prometheus_client import Counter, Gauge, Histogram, start_http_server
from sklearn.ensemble import IsolationForest

# Prometheus gauges for core risk metrics
latency_gauge = Gauge("trade_latency_ms", "Execution latency in milliseconds")
equity_gauge = Gauge("account_equity", "Current account equity")
var_gauge = Gauge("portfolio_var", "Estimated value-at-risk")
ws_ping_counter = Counter("ws_pings_total", "WebSocket pings sent")
ws_pong_counter = Counter("ws_pongs_total", "WebSocket pongs received")
ws_ping_rtt_histogram = Histogram(
    "ws_ping_rtt_seconds",
    "WebSocket ping-pong round trip time",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
)
listenkey_refresh_counter = Counter(
    "listenkey_refresh_total", "Binance listenKey refreshes"
)
ws_reconnect_counter = Counter("ws_reconnect_total", "WebSocket reconnects")
rate_limit_throttle_counter = Counter(
    "rate_limit_throttles_total", "Rate limiter throttles"
)


def start_metrics_server(port: int = 8000) -> None:
    """Start a Prometheus metrics HTTP server."""
    start_http_server(port)


def monitor_latency(value: float) -> None:
    """Update the latency gauge."""
    latency_gauge.set(value)


def monitor_equity(value: float) -> None:
    """Update the equity gauge."""
    equity_gauge.set(value)


def monitor_var(value: float) -> None:
    """Update the VaR gauge."""
    var_gauge.set(value)


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


__all__ = [
    "start_metrics_server",
    "monitor_latency",
    "monitor_equity",
    "monitor_var",
    "detect_anomalies",
    "latency_gauge",
    "equity_gauge",
    "var_gauge",
    "ws_ping_counter",
    "ws_pong_counter",
    "ws_ping_rtt_histogram",
    "listenkey_refresh_counter",
    "ws_reconnect_counter",
    "rate_limit_throttle_counter",
]
