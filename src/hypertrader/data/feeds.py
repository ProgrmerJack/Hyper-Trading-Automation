import asyncio
from threading import Thread
from typing import Any, Iterable

from cryptofeed import FeedHandler
from cryptofeed.defines import TRADES, TICKER
from cryptofeed.exchanges import Binance, Coinbase, Kraken

QUEUE: asyncio.Queue[Any] = asyncio.Queue(maxsize=10000)

def _cb(feed, pair, order):
    asyncio.create_task(QUEUE.put(order))

async def start_realtime_feed(symbols: Iterable[str] = ("BTC-USDT", "ETH-USDT")) -> None:
    fh = FeedHandler(config={"log": {"disabled": True}})
    fh.add_feed(Binance(symbols=symbols, channels=[TRADES, TICKER], callbacks={TRADES: _cb, TICKER: _cb}))
    fh.add_feed(Coinbase(symbols=[s.replace('-', '/') for s in symbols], channels=[TRADES], callbacks={TRADES: _cb}))
    fh.add_feed(Kraken(symbols=symbols, channels=[TRADES], callbacks={TRADES: _cb}))

    thread = Thread(target=fh.run, daemon=True)
    thread.start()
    try:
        await asyncio.Event().wait()
    finally:
        fh.stop()
        thread.join()
