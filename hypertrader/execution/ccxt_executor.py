"""Simple order execution using CCXT."""
from __future__ import annotations

import os
import os
from typing import Optional

import ccxt
from dotenv import load_dotenv

load_dotenv()


def _create_exchange(exchange_name: str) -> ccxt.Exchange:
    api_key = os.getenv(f"{exchange_name.upper()}_API_KEY")
    api_secret = os.getenv(f"{exchange_name.upper()}_API_SECRET")
    exchange_class = getattr(ccxt, exchange_name)
    return exchange_class({"apiKey": api_key, "secret": api_secret, "enableRateLimit": True})


def place_order(exchange_name: str, symbol: str, side: str, amount: float, order_type: str = "market", price: Optional[float] = None) -> dict:
    """Place an order via CCXT using API keys from environment."""
    exchange = _create_exchange(exchange_name)
    try:
        if order_type == "market":
            return exchange.create_market_order(symbol, side.lower(), amount)
        return exchange.create_limit_order(symbol, side.lower(), amount, price)
    finally:
        exchange.close()


def cancel_all(exchange_name: str, symbol: str) -> None:
    """Cancel all open orders for the given symbol."""
    exchange = _create_exchange(exchange_name)
    try:
        orders = exchange.fetch_open_orders(symbol)
        for order in orders:
            exchange.cancel_order(order["id"], symbol)
    finally:
        exchange.close()


def get_balance(exchange_name: str, asset: str = "USDT") -> float:
    """Return free balance for the specified asset."""
    exchange = _create_exchange(exchange_name)
    try:
        balance = exchange.fetch_balance()
        info = balance.get(asset, {})
        if isinstance(info, dict):
            return float(info.get("free", 0))
        return float(info)
    finally:
        exchange.close()


__all__ = ["place_order", "cancel_all", "get_balance"]
