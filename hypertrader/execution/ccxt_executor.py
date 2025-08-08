import asyncio
import logging
import os
import time

import ccxt.async_support as ccxt

from .validators import validate_order

ex = getattr(ccxt, os.getenv("EXCHANGE", "binance"))({
    "apiKey": os.getenv("API_KEY"),
    "secret": os.getenv("API_SECRET"),
    "enableRateLimit": True,
})


async def place_order(
    symbol: str, side: str, qty: float, price: float | None = None, params: dict | None = None
):
    """Place an order after validating venue limits."""

    params = params or {}

    await ex.load_markets()
    market = ex.market(symbol)
    if price is not None:
        if not validate_order(price, qty, market):
            raise ValueError("Order violates market limits")
        fn = ex.create_limit_buy_order if side == "buy" else ex.create_limit_sell_order
        before = time.perf_counter()
        order = await fn(symbol, qty, price, params)
    else:
        if not validate_order(0.0, qty, market):
            raise ValueError("Order violates market limits")
        fn = ex.create_market_buy_order if side == "buy" else ex.create_market_sell_order
        before = time.perf_counter()
        order = await fn(symbol, qty, params)
    latency_ms = (time.perf_counter() - before) * 1000
    logging.info("order-latency-ms %d", latency_ms)
    return order
