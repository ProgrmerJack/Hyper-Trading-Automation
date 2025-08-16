"""Lazy-loading package exports to avoid heavy import side effects."""

from importlib import import_module
from typing import Any

__all__ = [
    "run",
    "TradingOrchestrator",
    "calculate_position_size",
    "train_model",
    "ml_signal",
    "cross_validate_model",
    "connectors",
    "indicators",
    "strategies",
    "backtester",
    "utils",
    "bot",
]


def __getattr__(name: str) -> Any:  # pragma: no cover - thin wrapper
    if name == "run":
        return import_module("hypertrader.bot").run
    if name == "TradingOrchestrator":
        return import_module("hypertrader.orchestrator").TradingOrchestrator
    if name == "calculate_position_size":
        return import_module("hypertrader.utils.risk").calculate_position_size
    if name in {"train_model", "ml_signal", "cross_validate_model"}:
        mod = import_module("hypertrader.strategies.ml_strategy")
        return getattr(mod, name)
    raise AttributeError(name)

