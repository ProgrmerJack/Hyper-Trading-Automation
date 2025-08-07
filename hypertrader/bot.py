from __future__ import annotations
import json
import time
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

from .config import load_config
from .utils.sentiment import fetch_news_headlines, compute_sentiment_score
from .utils.features import (
    compute_atr,
    onchain_zscore,
    order_skew,
    dom_heatmap_ratio,
)
from .utils.risk import (
    calculate_position_size,
    dynamic_leverage,
    trailing_stop,
    drawdown_throttle,
    kill_switch,
    volatility_scaled_stop,
    ai_var,
    drl_throttle,
)
from .utils.monitoring import start_metrics_server, monitor_latency
from .utils.logging import get_logger, log_json
from .strategies.indicator_signals import generate_signal
from .strategies.ml_strategy import ml_signal

from .data.fetch_data import fetch_yahoo_ohlcv, fetch_order_book
from .data.onchain import fetch_eth_gas_fees
from .data.macro import (
    fetch_dxy,
    fetch_interest_rate,
    fetch_global_liquidity,
)
from .utils.macro import compute_macro_score

def run(
    symbol: str,
    account_balance: float = 10000.0,
    risk_percent: float = 2.0,
    news_api_key: str | None = None,
    fred_api_key: str | None = None,
    model_path: str | None = None,
    signal_path: str = "signal.json",
    config_path: str | None = None,
    state_path: str | None = None,
    exchange: str | None = None,
    etherscan_api_key: str | None = None,
) -> None:
    """Run one iteration of the trading pipeline.

    Fetches data, computes sentiment, generates a signal and writes it to JSON.
    """
    if config_path:
        cfg = load_config(config_path)
        symbol = cfg.get("trading", {}).get("symbol", symbol)
        account_balance = cfg.get("trading", {}).get("account_balance", account_balance)
        risk_percent = cfg.get("trading", {}).get("risk_percent", risk_percent)
        exchange = cfg.get("trading", {}).get("exchange", exchange)
        api_keys = cfg.get("api_keys", {})
        news_api_key = api_keys.get("news", news_api_key)
        fred_api_key = api_keys.get("fred", fred_api_key)
        etherscan_api_key = api_keys.get("etherscan", etherscan_api_key)

    logger = get_logger()
    start_time = time.time()
    try:
        start_metrics_server()
    except Exception as exc:
        log_json(logger, "metrics_server_failed", error=str(exc))

    # Determine location of persistent risk state
    if state_path is None:
        state_path = Path(signal_path).with_name("state.json")
    state_file = Path(state_path)
    state: dict[str, float] = {}
    if state_file.exists():
        try:
            state = json.loads(state_file.read_text())
        except json.JSONDecodeError:
            state = {}
    peak_equity = state.get("peak_equity", account_balance)
    drawdown = (peak_equity - account_balance) / peak_equity if peak_equity > 0 else 0.0
    allocation_factor = drawdown_throttle(account_balance, peak_equity)

    kill = kill_switch(drawdown)
    if kill:
        log_json(logger, "kill_switch_triggered", drawdown=drawdown)

    try:
        data = fetch_yahoo_ohlcv(symbol)
    except Exception as exc:
        log_json(logger, "data_fetch_failed", symbol=symbol, error=str(exc))
        return

    # On-chain metrics
    onchain_score = 0.0
    if etherscan_api_key is not None:
        try:
            gas_df = fetch_eth_gas_fees(etherscan_api_key)
            onchain_score = float(onchain_zscore(gas_df).iloc[-1])
        except Exception as exc:
            log_json(logger, "onchain_fetch_failed", error=str(exc))

    # Order book metrics
    book_skew = 0.0
    heatmap_ratio = 1.0
    if exchange:
        try:
            order_book = fetch_order_book(exchange, symbol.replace('-', '/'))
            book_skew = order_skew(order_book)
            heatmap_ratio = dom_heatmap_ratio(order_book)
        except Exception as exc:
            log_json(logger, "order_book_fetch_failed", error=str(exc))

    headlines: list[str] = []
    if news_api_key:
        try:
            headlines = fetch_news_headlines(news_api_key, query=symbol)
        except Exception as exc:
            log_json(logger, "news_fetch_failed", error=str(exc))
    sentiment = compute_sentiment_score(headlines)

    macro_score = 0.0
    if fred_api_key:
        try:
            dxy = fetch_dxy(api_key=fred_api_key)
            rates = fetch_interest_rate(fred_api_key)
            liquidity = fetch_global_liquidity(fred_api_key)
            macro_score = compute_macro_score(dxy, rates, liquidity)
        except Exception as exc:
            log_json(logger, "macro_fetch_failed", error=str(exc))
            macro_score = 0.0

    sig = generate_signal(
        data,
        sentiment,
        macro_score,
        onchain_score,
        book_skew,
        heatmap_ratio,
    )

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

    # Estimate recent volatility as std of returns
    volatility = float(data["close"].pct_change().rolling(10).std().iloc[-1])
    if pd.isna(volatility) or volatility <= 0:
        volatility = 0.02
    rl_factor = drl_throttle((drawdown, volatility))
    returns = data["close"].pct_change().dropna().tolist()
    var = ai_var(returns) if returns else 0.0
    allocation_factor *= rl_factor * max(0.1, 1 - var)
    leverage = 0.0
    if not kill:
        leverage = dynamic_leverage(
            account_balance, risk_percent * allocation_factor, volatility
        )

    if kill:
        sig.action = "HOLD"

    if sig.action == "BUY":
        stop_loss = volatility_scaled_stop(price, vix=volatility * 100, long=True)
        stop_loss = max(stop_loss, trailing_stop(price, price, atr))
        take_profit = price + 4 * atr
    elif sig.action == "SELL":
        stop_loss = volatility_scaled_stop(price, vix=volatility * 100, long=False)
        stop_loss = min(stop_loss, trailing_stop(price, price, atr))
        take_profit = price - 4 * atr
    else:
        stop_loss = None
        take_profit = None

    volume = 0.0
    if stop_loss is not None and not kill:
        volume = calculate_position_size(
            account_balance,
            risk_percent * allocation_factor,
            price,
            stop_loss,
        )
        volume *= leverage

    payload = {
        "action": sig.action,
        "volume": volume,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        # use timezone-aware UTC timestamp to avoid deprecation warnings
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "leverage": leverage,
        "var": var,
    }
    Path(signal_path).write_text(json.dumps(payload))

    latency = time.time() - start_time
    monitor_latency(latency)
    log_json(
        logger,
        "signal_generated",
        symbol=symbol,
        action=sig.action,
        price=float(price),
        latency=latency,
        slippage=0.0,
        leverage=leverage,
        drawdown=drawdown,
        var=var,
    )

    state["peak_equity"] = max(peak_equity, account_balance)
    state["equity"] = account_balance
    state_file.write_text(json.dumps(state))


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
    parser.add_argument("--config")
    parser.add_argument("--state_path")
    parser.add_argument("--exchange")
    parser.add_argument("--etherscan_api_key")
    args = parser.parse_args()

    run(
        args.symbol,
        account_balance=args.account_balance,
        risk_percent=args.risk_percent,
        news_api_key=args.news_api_key,
        fred_api_key=args.fred_api_key,
        model_path=args.model_path,
        signal_path=args.signal_path,
        config_path=args.config,
        state_path=args.state_path,
        exchange=args.exchange,
        etherscan_api_key=args.etherscan_api_key,
    )

