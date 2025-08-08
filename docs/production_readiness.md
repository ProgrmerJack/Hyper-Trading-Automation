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
1. Make WebSocket ingestion the default and instrument latency metrics (tick→decision, decision→ack, ack→fill).
2. Implement a persistent OMS with cancel-on-disconnect and reduce-only/post-only enforcement.
3. Remove the file-based MT5 bridge and execute directly against exchange APIs or a dedicated execution service.
4. Upgrade the backtesting engine to model depth, queue position, latency and fees.
5. Purge Yahoo Finance from live decision paths; use exchange APIs or paid feeds for both live and historical data.
6. Tune rate limits and reconnection logic per venue with explicit tests.
7. Add fee/funding-aware gating to avoid trades with negative expected edge.

## Deployment checklist
- **Paper / shadow trading**: verify WebSocket stability, simulate OMS flows, and test circuit breakers.
- **Small live trading**: enable post-only where possible, enforce exposure limits and a kill-switch.
- **Scale up** only after demonstrating stability over at least 30 sessions per venue and symbol.

These steps will help evolve the project from a prototype into a robust automation system.
