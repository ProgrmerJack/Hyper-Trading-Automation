"""Configuration loading utilities for Hyper-Trading Automation.

This module loads YAML configuration files that store API keys, trading
symbols, risk parameters and backtest settings. A default `config.yaml`
at the project root is used when no path is provided.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml
from pydantic import BaseModel, Field, ValidationError


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


class TradingConfig(BaseModel):
    """Schema for trading related settings."""

    symbol: str
    account_balance: float = Field(100.0, ge=0)
    risk_percent: float = Field(..., ge=0, le=100)
    max_exposure: float = Field(3.0, ge=0)


class APIKeys(BaseModel):
    """Schema for API keys."""

    fred: str | None = None
    news: str | None = None
    etherscan: str | None = None


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
        if hasattr(ConfigModel, "model_validate"):
            model = ConfigModel.model_validate(data)  # type: ignore[attr-defined]
        else:  # pydantic v1 fallback
            model = ConfigModel.parse_obj(data)
    except ValidationError as exc:  # pragma: no cover - simple pass through
        raise ValueError(f"Invalid configuration: {exc}") from exc

    # Fill API keys from environment variables if not provided in config
    api = model.api_keys or APIKeys()
    api.fred = api.fred or os.getenv("FRED_API_KEY")
    api.news = api.news or os.getenv("NEWS_API_KEY")
    api.etherscan = api.etherscan or os.getenv("ETHERSCAN_API_KEY")
    model.api_keys = api

    return model.model_dump() if hasattr(model, "model_dump") else model.dict()


__all__ = ["load_config", "DEFAULT_CONFIG_PATH"]

