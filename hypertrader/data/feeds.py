import asyncio
import logging
from cryptofeed import FeedHandler
from cryptofeed.defines import TRADES, TICKER
from cryptofeed.exchanges import Binance, Coinbase, Kraken

QUEUE = asyncio.Queue(maxsize=10000)

def _cb(feed, pair, order):
    asyncio.create_task(QUEUE.put(order))

def start_realtime_feed(symbols=("BTC-USDT", "ETH-USDT")):
    fh = FeedHandler(config={"log": {"disabled": True}})
    fh.add_feed(Binance(symbols=symbols, channels=[TRADES, TICKER], callbacks={TRADES: _cb, TICKER: _cb}))
    fh.add_feed(Coinbase(symbols=[s.replace('-', '/') for s in symbols], channels=[TRADES], callbacks={TRADES: _cb}))
    fh.add_feed(Kraken(symbols=symbols, channels=[TRADES], callbacks={TRADES: _cb}))
    fh.run()
