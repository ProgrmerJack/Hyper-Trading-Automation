from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

from .utils.sentiment import fetch_news_headlines, compute_sentiment_score
from .utils.features import compute_atr
from .utils.risk import calculate_position_size
from .strategies.indicator_signals import generate_signal
from .strategies.ml_strategy import ml_signal

from .data.macro import (
    fetch_dxy,
    fetch_interest_rate,
    fetch_global_liquidity,
)
from .utils.macro import compute_macro_score

def fetch_yahoo_ohlcv(symbol: str, interval: str = "1h", lookback: str = "7d") -> pd.DataFrame:
    """Fetch OHLCV data from Yahoo Finance using yfinance."""
    df = yf.download(symbol, period=lookback, interval=interval, progress=False)
    if df.empty:
        raise ValueError("No data fetched from Yahoo Finance")
    df.index.name = "timestamp"
    df.rename(columns={c: c.lower() for c in df.columns}, inplace=True)
    return df[["open", "high", "low", "close", "volume"]]


def run(
    symbol: str,
    account_balance: float = 10000.0,
    risk_percent: float = 2.0,
    news_api_key: str | None = None,
    fred_api_key: str | None = None,
    model_path: str | None = None,
    signal_path: str = "signal.json",
) -> None:
    """Run one iteration of the trading pipeline.

    Fetches data, computes sentiment, generates a signal and writes it to JSON.
    """
    data = fetch_yahoo_ohlcv(symbol)
    headlines = []
    if news_api_key:
        headlines = fetch_news_headlines(news_api_key, query=symbol)
    sentiment = compute_sentiment_score(headlines)

    macro_score = 0.0
    if fred_api_key:
        try:
            dxy = fetch_dxy()
            rates = fetch_interest_rate(fred_api_key)
            liquidity = fetch_global_liquidity(fred_api_key)
            macro_score = compute_macro_score(dxy, rates, liquidity)
        except Exception:
            macro_score = 0.0

    sig = generate_signal(data, sentiment, macro_score)

    if model_path:
        try:
            model = pd.read_pickle(model_path)
            ml_sig = ml_signal(model, data)
            if ml_sig.action != sig.action:
                # Require agreement for trade
                sig.action = "HOLD"
        except FileNotFoundError:
            pass


    price = data["close"].iloc[-1]
    atr = compute_atr(data).iloc[-1]
    if sig.action == "BUY":
        stop_loss = price - 2 * atr
        take_profit = price + 4 * atr
    elif sig.action == "SELL":
        stop_loss = price + 2 * atr
        take_profit = price - 4 * atr
    else:
        stop_loss = None
        take_profit = None

    volume = 0.0
    if stop_loss is not None:
        volume = calculate_position_size(account_balance, risk_percent, price, stop_loss)

    payload = {
        "action": sig.action,
        "volume": volume,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        # use timezone-aware UTC timestamp to avoid deprecation warnings
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    Path(signal_path).write_text(json.dumps(payload))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run autonomous trading bot")
    parser.add_argument("symbol", help="Trading pair symbol e.g. BTC-USD")
    parser.add_argument("--account_balance", type=float, default=10000.0)
    parser.add_argument("--risk_percent", type=float, default=2.0)
    parser.add_argument("--news_api_key")
    parser.add_argument("--fred_api_key")
    parser.add_argument("--model_path")

    parser.add_argument("--signal_path", default="signal.json")
    args = parser.parse_args()

    run(
        args.symbol,
        account_balance=args.account_balance,
        risk_percent=args.risk_percent,
        news_api_key=args.news_api_key,
        fred_api_key=args.fred_api_key,
        model_path=args.model_path,
        signal_path=args.signal_path,
    )

