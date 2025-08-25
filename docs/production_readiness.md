# Production Readiness Recommendations

This repository is a research prototype and is **not** production-ready for live or high-frequency trading. The checklist below captures the major gaps that must be addressed before considering real-money deployment.

## Architecture weaknesses

- Polling-first pipeline with optional WebSocket support. Production systems should use WebSockets end-to-end, feeding an event-driven signal generator and order management system (OMS).
- MT5 bridge communicates via `signal.json`; replace file I/O with direct exchange APIs or a persistent IPC channel.
- No hardened OMS. Implement idempotent `clientOrderId`, state transitions (`NEW`, `ACK`, `PARTIAL`, `FILLED`, `CANCELED`), retries and persistence for crash recovery.

## Risk-control gaps

- Basic pre-send validation for venue rules (minimum notional and step size) has been added, but post-only and reduce-only flags remain unhandled.
- Fee and slippage-aware edge checks implemented via ``RiskManager``; still need latency/slippage circuit breakers.
- Circuit breakers: daily loss caps and per-symbol exposure limits exist, yet cancel-all on disconnect and latency breakers are still missing.

## Backtesting limitations

- Vectorized backtests do not model order book microstructure or latency. Use an event-driven simulator with depth, queue and realistic fills.
- Historical data should come from exchange-grade sources; Yahoo Finance has been removed from the live code path.
- Include actual fees and funding costs in simulations.

## Prioritized fixes

1. Make WebSocket ingestion the default and instrument latency metrics (tick→decision, decision→ack, ack→fill). [Done: baseline WS feed with reconnects, metrics, alerts]
2. Implement a persistent OMS with cancel-on-disconnect and reduce-only/post-only enforcement. [Done: SQLite OMSStore, cancel-all on disconnect, post/reduce flags]
3. Remove the file-based MT5 bridge and execute directly against exchange APIs or a dedicated execution service. [In progress]
4. Upgrade the backtesting engine to model depth, queue position, latency and fees. [Open]
5. Purge Yahoo Finance from live decision paths; use exchange APIs or paid feeds for both live and historical data. [Done]

## Windows 11 24/7 deployment notes

- Disable sleep: Control Panel → Power Options → High performance; never sleep; disable USB selective suspend.

- Create conda env: `conda create -n trading-py310 python=3.10 -y && conda activate trading-py310 && pip install -r requirements.txt`.

- Environment: create `.env` (based on `examples/.env.example` content in README) with `API_KEY`, `API_SECRET`, `EXCHANGE=binance`.

- Service: Use Task Scheduler (On startup, run `python -m hypertrader.bot --config config.yaml`) or NSSM to run `python scripts/connectivity_check.py` then `python -m hypertrader.bot ...`.

- Logging: set `LOG_FILE`, `LOG_MAX_BYTES`, `LOG_BACKUP_COUNT`; redirect stdout to file; collect with Prometheus at `METRICS_PORT`.

- Alerts: set `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` for critical failures (WS disconnects, cancel-all, latency hard breaches, kill switch).

- Crash recovery: OMSStore persists `orders`, `fills`, `positions`; on start the bot reconciles open orders and positions; private WS keeps state in sync.

1. Tune rate limits and reconnection logic per venue with explicit tests.
2. Add fee/funding-aware gating to avoid trades with negative expected edge.

## Deployment checklist

- **Paper / shadow trading**: verify WebSocket stability, simulate OMS flows, and test circuit breakers.
- **Small live trading**: enable post-only where possible, enforce exposure limits and a kill-switch.
- **Scale up** only after demonstrating stability over at least 30 sessions per venue and symbol.

These steps will help evolve the project from a prototype into a robust automation system.
