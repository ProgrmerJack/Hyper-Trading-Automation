import asyncio
import logging
import os
import time

import ccxt.async_support as ccxt

ex = getattr(ccxt, os.getenv("EXCHANGE", "binance"))({
    "apiKey": os.getenv("API_KEY"),
    "secret": os.getenv("API_SECRET"),
    "enableRateLimit": True,
})


async def place_order(symbol: str, side: str, qty: float, price: float | None = None, params: dict | None = None):
    params = params or {}
    fn = ex.create_limit_buy_order if side == "buy" else ex.create_limit_sell_order
    before = time.perf_counter()
    order = await fn(symbol, qty, price, params)
    latency_ms = (time.perf_counter() - before) * 1000
    logging.info("order-latency-ms %d", latency_ms)
    return order
