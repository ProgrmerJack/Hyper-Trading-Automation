import logging
import os
import time
import uuid

import ccxt.async_support as ccxt

from .validators import validate_order

ex = getattr(ccxt, os.getenv("EXCHANGE", "binance"))({
    "apiKey": os.getenv("API_KEY"),
    "secret": os.getenv("API_SECRET"),
    "enableRateLimit": True,
})


async def place_order(
    symbol: str,
    side: str,
    qty: float,
    price: float | None = None,
    client_id: str | None = None,
    params: dict | None = None,
    post_only: bool = False,
    reduce_only: bool = False,
):
    """Place an order after validating venue limits.

    Parameters
    ----------
    symbol : str
        Trading pair symbol e.g. ``BTC/USDT``.
    side : str
        Either ``buy`` or ``sell``.
    qty : float
        Quantity to trade.
    price : float | None, optional
        Limit price.  If ``None`` a market order is sent.
    client_id : str | None, optional
        Optional idempotency key.  A random UUID will be generated if not provided.
    params : dict | None, optional
        Additional parameters for the CCXT order call.
    post_only : bool, optional
        When ``True`` the order is submitted with a post-only flag.  Raises
        ``RuntimeError`` if the venue does not support it.
    reduce_only : bool, optional
        When ``True`` the order is submitted with a reduce-only flag.  Raises
        ``RuntimeError`` if unsupported by the venue.
    """

    params = params.copy() if params else {}
    if client_id is None:
        client_id = uuid.uuid4().hex
    params.setdefault("clientOrderId", client_id)

    if post_only:
        if not ex.has.get("createPostOnlyOrder", False):
            raise RuntimeError("post-only orders not supported")
        params["postOnly"] = True
    if reduce_only:
        if not ex.has.get("createReduceOnlyOrder", False):
            raise RuntimeError("reduce-only orders not supported")
        params["reduceOnly"] = True

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


async def cancel_order(symbol: str, order_id: str):
    """Cancel a specific order by id."""

    await ex.load_markets()
    return await ex.cancel_order(order_id, symbol)
