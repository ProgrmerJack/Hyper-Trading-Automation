from __future__ import annotations
import asyncio
import json
import time
import os
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Any
from collections.abc import Sequence
from collections import deque
from dataclasses import dataclass, field

import pandas as pd
import numpy as np
from dotenv import load_dotenv

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
    quantum_leverage_modifier,
    cap_position_value,
)
from .utils.volatility import rank_symbols_by_volatility
from .utils.monitoring import (
    start_metrics_server,
    monitor_latency,
    monitor_equity,
    monitor_var,
    detect_anomalies,
)
from .utils.logging import get_logger, log_json
from .strategies.indicator_signals import generate_signal
from .strategies.ml_strategy import ml_signal

from .data.fetch_data import fetch_ohlcv, fetch_order_book
from .data.oms_store import OMSStore
from .data.onchain import fetch_eth_gas_fees
from .data.macro import (
    fetch_dxy,
    fetch_interest_rate,
    fetch_global_liquidity,
)
from .utils.macro import compute_macro_score
from .execution.ccxt_executor import place_order, cancel_order, ex
from .risk.manager import RiskManager, RiskParams

load_dotenv()


@dataclass
class TradingBot:
    """High‑level trading bot orchestrating data, strategies and execution.

    Parameters
    ----------
    connector : ExchangeConnector
        Interface to market data and order execution (live or simulation).
    strategy : object
        Strategy or meta strategy instance that produces orders.
    symbol : str
        Trading pair to operate on.  For multi‑symbol trading,
        instantiate separate bots or extend the class to handle
        multiple symbols.
    base_order_size : float, optional
        Base quantity used when strategies do not specify sizes.
    max_drawdown : float, optional
        Maximum tolerated drawdown expressed as fraction of equity
        (e.g., 0.1 for 10%).  Used by the drawdown throttle.
    stop_loss_pct : float, optional
        Percentage for trailing stop on open positions.  Default is
        0.02 (2%).
    """

    connector: Any
    strategy: Any
    symbol: str
    base_order_size: float = 1.0
    max_drawdown: float = 0.1
    stop_loss_pct: float = 0.02
    # internal state
    equity: float = 0.0
    peak_equity: float = 0.0
    open_position: float = 0.0
    last_price: float = 0.0
    order_history: list[tuple[str, float, float, datetime]] = field(default_factory=list, init=False)

    def update_equity(self, price: float) -> None:
        """Update equity and peak equity based on current price and position."""
        self.last_price = price
        self.equity = self.open_position * price
        self.peak_equity = max(self.peak_equity, self.equity)

    def on_new_tick(self, price: float, trades: list[dict]) -> None:
        """Process a new market tick.

        Parameters
        ----------
        price : float
            Latest trade or mid price of the instrument.
        trades : list of dict
            Recent trades used for toxicity and regime analysis.
        """
        from .utils.features import flow_toxicity, detect_entropy_regime
        from .utils.rl_utils import dynamic_order_size
        
        # Update internal equity
        self.update_equity(price)
        # Compute microstructure signals for RL sizing
        toxicity = flow_toxicity(trades, window=min(len(trades), 100))
        # Determine regime from price directions (use last 20 order history directions)
        directions = [1 if self.order_history[i][1] > 0 else 0 for i in range(max(0, len(self.order_history) - 20), len(self.order_history))] if self.order_history else []
        regime = detect_entropy_regime(directions) if directions else "normal"
        # Generate orders from strategy
        if hasattr(self.strategy, "update"):
            orders = self.strategy.update(price)  # pass price only to simple strategies
        else:
            orders = []
        # Risk checks: trailing stop and drawdown
        # If we have an open position, apply trailing stop
        stop_price = trailing_stop(self.open_position, self.last_price, self.stop_loss_pct)
        if self.open_position > 0 and price < stop_price:
            # Sell to close long position
            orders.append(("sell", price, self.open_position))
            self.open_position = 0.0
        elif self.open_position < 0 and price > stop_price:
            # Buy to close short position
            orders.append(("buy", price, -self.open_position))
            self.open_position = 0.0
        # Drawdown throttle: skip orders if equity falls too far from peak
        if drawdown_throttle(self.equity, self.peak_equity, self.max_drawdown):
            orders = []
        # Process orders
        for side, order_price, qty in orders:
            # Determine dynamic size based on RL sizing
            size = dynamic_order_size(0.6 if side == "buy" else 0.4, toxicity, regime, self.base_order_size)
            # If size or qty is zero, skip
            if size <= 0.0 or qty <= 0.0:
                continue
            # Use price if provided
            exec_price = order_price or price
            # Send order to connector
            self.connector.place_order(self.symbol, side, size, exec_price)
            # Update position
            if side == "buy":
                self.open_position += size
            else:
                self.open_position -= size
            # Record order
            self.order_history.append((side, size, exec_price, datetime.now(timezone.utc)))

async def _run(
    symbol: str | Sequence[str],
    account_balance: float = 100.0,
    risk_percent: float = 5.0,
    news_api_key: str | None = None,
    fred_api_key: str | None = None,
    model_path: str | None = None,
    signal_path: str = "signal.json",
    config_path: str | None = None,
    state_path: str | Path | None = None,
    exchange: str | None = None,
    etherscan_api_key: str | None = None,
    max_exposure: float = 3.0,
    live: bool = False,
    data: pd.DataFrame | None = None,
    store: OMSStore | None = None,
) -> None:
    """Run one iteration of the trading pipeline.

    Parameters
    ----------
    symbol:
        Either a single ticker symbol or a sequence of symbols.  When
        multiple symbols are provided the bot selects the one with the
        highest recent volatility.

    Fetches data, computes sentiment, generates a signal and writes it to JSON.
    """
    if config_path:
        cfg = load_config(config_path)
        symbol = cfg.get("trading", {}).get("symbol", symbol)
        account_balance = cfg.get("trading", {}).get("account_balance", account_balance)
        risk_percent = cfg.get("trading", {}).get("risk_percent", risk_percent)
        exchange = cfg.get("trading", {}).get("exchange", exchange)
        max_exposure = cfg.get("trading", {}).get("max_exposure", max_exposure)
        api_keys = cfg.get("api_keys", {})
        news_api_key = api_keys.get("news", news_api_key)
        fred_api_key = api_keys.get("fred", fred_api_key)
        etherscan_api_key = api_keys.get("etherscan", etherscan_api_key)

    # fall back to environment variables for API keys
    news_api_key = news_api_key or os.getenv("NEWS_API_KEY")
    fred_api_key = fred_api_key or os.getenv("FRED_API_KEY")
    etherscan_api_key = etherscan_api_key or os.getenv("ETHERSCAN_API_KEY")

    # If a sequence of symbols is provided, pick the most volatile one
    if isinstance(symbol, Sequence) and not isinstance(symbol, str):
        try:
            ranked = rank_symbols_by_volatility(symbol)
            symbol = ranked[0] if ranked else list(symbol)[0]
        except Exception:
            # fall back to first symbol if ranking fails
            symbol = list(symbol)[0]

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
    state: dict[str, Any] = {}
    if state_file.exists():
        try:
            state = json.loads(state_file.read_text())
        except json.JSONDecodeError:
            state = {}
    latencies = deque(state.get("latencies", []), maxlen=120)
    latency_breach = state.get("latency_breach", 0)
    peak_equity = state.get("peak_equity", account_balance)
    drawdown = (peak_equity - account_balance) / peak_equity if peak_equity > 0 else 0.0
    allocation_factor = drawdown_throttle(account_balance, peak_equity)
    p95_latency = 0.0
    if len(latencies) >= 20:
        p95_latency = float(np.percentile(list(latencies), 95))
        if p95_latency > 2.0:
            latency_breach += 1
        else:
            latency_breach = 0
    latency_soft = latency_breach > 0
    latency_hard = latency_breach >= 3
    if store is None:
        store = OMSStore(state_file.with_suffix(".db"))
        owns_store = True
    else:
        owns_store = False
    open_orders: dict[str, Any] = {
        oid: {"symbol": sym, "side": side, "volume": vol}
        for oid, sym, side, vol, _ in await store.fetch_open_orders()
    }

    risk_cfg = cfg.get("risk", {}) if config_path else {}
    params = RiskParams(
        max_daily_loss=risk_cfg.get("max_daily_loss", account_balance * 0.2),
        max_position=risk_cfg.get("max_position", account_balance * max_exposure),
        fee_rate=risk_cfg.get("fee_rate", 0.0),
        slippage=risk_cfg.get("slippage", 0.0),
        symbol_limits=risk_cfg.get("symbol_limits"),
        max_var=risk_cfg.get("max_var"),
        max_volatility=risk_cfg.get("max_volatility"),
    )
    risk_manager = RiskManager(params)
    risk_manager.reset_day(account_balance)

    ccxt_symbol = symbol.replace('-', '/')

    async def reconcile() -> None:
        try:
            remote_orders = await ex.fetch_open_orders(ccxt_symbol)
        except Exception:
            remote_orders = []
        remote_ids = {o.get("clientOrderId") or o.get("id") for o in remote_orders}
        for oid in list(open_orders):
            if oid not in remote_ids:
                await store.remove_order(oid)
                del open_orders[oid]
        for o in remote_orders:
            cid = o.get("clientOrderId") or o.get("id")
            if cid and cid not in open_orders:
                await store.record_order(
                    cid,
                    o.get("clientOrderId"),
                    o.get("symbol", ccxt_symbol),
                    o.get("side", ""),
                    float(o.get("amount") or o.get("remaining") or 0.0),
                    o.get("price"),
                    o.get("status", "open"),
                    (o.get("timestamp") or 0) / 1000,
                )
                open_orders[cid] = {
                    "symbol": o.get("symbol", ccxt_symbol),
                    "side": o.get("side", ""),
                    "volume": float(o.get("amount") or o.get("remaining") or 0.0),
                }
        try:
            positions = await ex.fetch_positions([ccxt_symbol])
            for p in positions:
                qty = float(
                    p.get("contracts")
                    or p.get("positionAmt")
                    or p.get("size")
                    or 0.0
                )
                if qty:
                    await store.upsert_position(
                        p.get("symbol", ccxt_symbol),
                        qty,
                        float(p.get("entryPrice") or 0.0),
                        float(p.get("liquidationPrice") or 0.0),
                        time.time(),
                    )
        except Exception:
            pass

    # reconcile open orders and positions with the exchange
    if live and exchange:
        await reconcile()
        if time.time() - state.get("last_reconcile", 0) > 300:
            await reconcile()
            state["last_reconcile"] = time.time()

    # cancel any lingering open orders from previous session
    if live and exchange and open_orders:
        for oid, info in list(open_orders.items()):
            try:
                await cancel_order(info["symbol"], oid)
                await store.remove_order(oid)
                del open_orders[oid]
            except Exception as exc:
                log_json(logger, "cancel_failed", order_id=oid, error=str(exc))

    kill = kill_switch(drawdown)
    if kill:
        log_json(logger, "kill_switch_triggered", drawdown=drawdown)
    tasks: list[asyncio.Future] = []
    keys: list[str] = []
    if data is None:
        tasks.append(asyncio.to_thread(fetch_ohlcv, exchange or "binance", ccxt_symbol, "1m"))
        keys.append("data")
    if etherscan_api_key:
        tasks.append(asyncio.to_thread(fetch_eth_gas_fees, etherscan_api_key))
        keys.append("gas")
    if exchange:
        tasks.append(asyncio.to_thread(fetch_order_book, exchange, ccxt_symbol))
        keys.append("order_book")
    if news_api_key:
        tasks.append(asyncio.to_thread(fetch_news_headlines, news_api_key, query=symbol))
        keys.append("news")
    if fred_api_key:
        tasks.append(asyncio.to_thread(fetch_dxy, api_key=fred_api_key))
        keys.append("dxy")
        tasks.append(asyncio.to_thread(fetch_interest_rate, fred_api_key))
        keys.append("rates")
        tasks.append(asyncio.to_thread(fetch_global_liquidity, fred_api_key))
        keys.append("liquidity")

    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        result_map = dict(zip(keys, results))
    else:
        result_map = {}

    if data is None:
        data = result_map.get("data")
        if isinstance(data, Exception) or data is None:
            log_json(logger, "data_fetch_failed", symbol=symbol, error=str(data))
            return

    onchain_score = 0.0
    gas_df = result_map.get("gas")
    if isinstance(gas_df, Exception):
        log_json(logger, "onchain_fetch_failed", error=str(gas_df))
    elif gas_df is not None:
        onchain_score = float(onchain_zscore(gas_df).iloc[-1])

    book_skew = 0.0
    heatmap_ratio = 1.0
    order_book = result_map.get("order_book")
    if isinstance(order_book, Exception):
        log_json(logger, "order_book_fetch_failed", error=str(order_book))
    elif order_book is not None:
        book_skew = order_skew(order_book)
        heatmap_ratio = dom_heatmap_ratio(order_book)

    headlines: list[str] = []
    news = result_map.get("news")
    if isinstance(news, Exception):
        log_json(logger, "news_fetch_failed", error=str(news))
    elif news is not None:
        headlines = news
    sentiment = compute_sentiment_score(headlines)

    macro_score = 0.0
    if fred_api_key:
        dxy = result_map.get("dxy")
        rates = result_map.get("rates")
        liquidity = result_map.get("liquidity")
        if any(isinstance(r, Exception) for r in (dxy, rates, liquidity)):
            err = ";".join(str(r) for r in (dxy, rates, liquidity) if isinstance(r, Exception))
            log_json(logger, "macro_fetch_failed", error=err)
        else:
            macro_score = compute_macro_score(dxy, rates, liquidity)

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
        q_factor = quantum_leverage_modifier([drawdown, volatility])
        leverage *= max(0.1, q_factor)

    if kill:
        sig.action = "HOLD"

    if params.max_var and var > params.max_var:
        sig.action = "HOLD"
    if params.max_volatility and volatility > params.max_volatility:
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
        volume = cap_position_value(volume, price, account_balance, max_exposure)

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
    position_value = volume * price
    edge = abs((take_profit - price) / price) if take_profit else 0.0
    if latency_hard and live and exchange:
        log_json(logger, "latency_slo_triggered", stage="hard", p95=p95_latency)
        for oid, info in list(open_orders.items()):
            try:
                await cancel_order(info["symbol"], oid)
                await store.update_order_status(oid, "CANCELED")
                del open_orders[oid]
            except Exception:
                pass
        sig.action = "HOLD"
    elif latency_soft and live:
        log_json(logger, "latency_slo_triggered", stage="soft", p95=p95_latency)
        sig.action = "HOLD"
    if live and exchange and sig.action != "HOLD" and volume > 0:
        if risk_manager.check_order(account_balance, ccxt_symbol, position_value, edge):
            client_id = uuid.uuid4().hex
            try:
                order_resp = await place_order(
                    ccxt_symbol, sig.action, volume, client_id=client_id
                )
                open_orders[client_id] = {
                    "symbol": ccxt_symbol,
                    "side": sig.action,
                    "volume": volume,
                }
                await store.record_order(
                    client_id,
                    client_id,
                    ccxt_symbol,
                    sig.action,
                    volume,
                    order_resp.get("price"),
                    order_resp.get("status", "open"),
                    time.time(),
                )
                payload["client_order_id"] = client_id
            except Exception as exc:
                log_json(logger, "order_failed", error=str(exc))
        else:
            log_json(logger, "risk_check_failed", symbol=symbol, position_value=position_value)

    else:
        Path(signal_path).write_text(json.dumps(payload))
    latency = time.time() - start_time
    monitor_latency(latency)
    monitor_equity(account_balance)
    monitor_var(var)

    latencies.append(latency)
    if len(latencies) > 5:
        labels = detect_anomalies(latencies)
        if labels[-1] == -1:
            log_json(logger, "latency_anomaly", latency=latency)

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
    state["latencies"] = list(latencies)
    state["latency_breach"] = latency_breach
    state_file.write_text(json.dumps(state))
    if store and owns_store:
        await store.close()


def run(
    symbol: str | Sequence[str],
    account_balance: float = 100.0,
    risk_percent: float = 5.0,
    news_api_key: str | None = None,
    fred_api_key: str | None = None,
    model_path: str | None = None,
    signal_path: str = "signal.json",
    config_path: str | None = None,
    state_path: str | Path | None = None,
    exchange: str | None = None,
    etherscan_api_key: str | None = None,
    max_exposure: float = 3.0,
    live: bool = False,
) -> None:
    """Synchronous wrapper that executes the async trading pipeline."""
    asyncio.run(
        _run(
            symbol,
            account_balance=account_balance,
            risk_percent=risk_percent,
            news_api_key=news_api_key,
            fred_api_key=fred_api_key,
            model_path=model_path,
            signal_path=signal_path,
            config_path=config_path,
            state_path=state_path,
            exchange=exchange,
            etherscan_api_key=etherscan_api_key,
            max_exposure=max_exposure,
            live=live,
        )
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run autonomous trading bot")
    parser.add_argument(
        "symbol",
        nargs="+",
        help="One or more trading pair symbols e.g. BTC-USD ETH-USD",
    )
    parser.add_argument("--account_balance", type=float, default=100.0)
    parser.add_argument("--risk_percent", type=float, default=5.0)
    parser.add_argument("--news_api_key")
    parser.add_argument("--fred_api_key")
    parser.add_argument("--model_path")

    parser.add_argument("--signal_path", default="signal.json")
    parser.add_argument("--config")
    parser.add_argument("--state_path")
    parser.add_argument("--exchange")
    parser.add_argument("--etherscan_api_key")
    parser.add_argument("--max_exposure", type=float, default=3.0)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()

    run(
        args.symbol if len(args.symbol) > 1 else args.symbol[0],
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
        max_exposure=args.max_exposure,
        live=args.live,
    )

