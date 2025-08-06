"""Configuration loading utilities for Hyper-Trading Automation.

This module loads YAML configuration files that store API keys, trading
symbols, risk parameters and backtest settings. A default `config.yaml`
at the project root is used when no path is provided.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    """Load configuration from a YAML file.

    Parameters
    ----------
    path : str | Path | None
        Optional path to a YAML file. If ``None`` the default project level
        ``config.yaml`` is used.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """

    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open() as f:
        return yaml.safe_load(f)


__all__ = ["load_config", "DEFAULT_CONFIG_PATH"]

