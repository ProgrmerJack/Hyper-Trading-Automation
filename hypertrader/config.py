"""Configuration loading utilities for Hyper-Trading Automation.

This module loads YAML configuration files that store API keys, trading
symbols, risk parameters and backtest settings. A default `config.yaml`
at the project root is used when no path is provided.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml
from pydantic import BaseModel, Field, ValidationError


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


class TradingConfig(BaseModel):
    """Schema for trading related settings."""

    symbol: str
    risk_percent: float = Field(..., ge=0, le=100)


class APIKeys(BaseModel):
    """Schema for API keys."""

    fred: str | None = None


class ConfigModel(BaseModel):
    """Top-level configuration schema."""

    api_keys: APIKeys | None = None
    trading: TradingConfig


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    """Load configuration from a YAML file with validation.

    Parameters
    ----------
    path : str | Path | None
        Optional path to a YAML file. If ``None`` the default project level
        ``config.yaml`` is used.

    Returns
    -------
    dict
        Parsed configuration dictionary validated against :class:`ConfigModel`.
    """

    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open() as f:
        data = yaml.safe_load(f)
    try:
        model = ConfigModel.model_validate(data)
    except ValidationError as exc:  # pragma: no cover - simple pass through
        raise ValueError(f"Invalid configuration: {exc}") from exc
    return model.model_dump()


__all__ = ["load_config", "DEFAULT_CONFIG_PATH"]

