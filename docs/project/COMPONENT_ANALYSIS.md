# HyperTrader Component Analysis Report

## Component Count Analysis

### 1. Strategies (21 components)
**Enabled by default:**
- indicator (base signal)
- ma_cross, rsi, bb, macd, ichimoku, psar, cci, keltner, fibonacci (9)
- ml_simple, market_maker, stat_arb, triangular_arb, event_trading, rl_strategy (6)
- spot_grid, futures_grid, rebalancing, spot_dca, funding_arb (5)

**Available but not enabled by default:**
- donchian, mean_reversion, momentum (3)

### 2. Technical Indicators (20+ components)
**From utils.features:**
- compute_atr, compute_exchange_netflow, compute_twap, compute_cumulative_delta
- compute_moving_average, compute_ichimoku, compute_parabolic_sar
- compute_keltner_channels, compute_cci, compute_fibonacci_retracements

**From indicators.microstructure:**
- compute_microprice, flow_toxicity, detect_iceberg, detect_entropy_regime

**Additional technical indicators computed:**
- SMA_20, volatility, RSI, Bollinger Bands, MACD, Stochastic, ADX

### 3. Sentiment Analyzers (8+ components)
**Original sentiment:**
- VADER sentiment (compute_sentiment_score)

**New ML sentiment (4):**
- FinBERT financial sentiment
- Twitter RoBERTa social sentiment  
- BART-MNLI catalyst classification
- Regime forecasting

**Macro sentiment (3):**
- DXY analysis, Interest rates, Global liquidity

**Onchain sentiment (1):**
- Gas fee z-score analysis

### 4. Risk Management Components (15+ components)
- calculate_position_size, dynamic_leverage, trailing_stop
- drawdown_throttle, kill_switch, volatility_scaled_stop
- ai_var, drl_throttle, quantum_leverage_modifier
- cap_position_value, fee_slippage_gate, compound_capital, shap_explain

### 5. Microstructure Components (10+ components)
- order_skew, dom_heatmap_ratio, microprice
- iceberg detection, entropy regime detection
- flow toxicity, book_skew, heatmap_ratio

### 6. Data Sources (8+ components)
- OHLCV data, Order book data, News headlines
- ETH gas fees, DXY, Interest rates, Global liquidity
- Exchange netflow data

### 7. Execution & Monitoring (8+ components)
- Order placement, cancellation, position tracking
- Latency monitoring, equity monitoring, VAR monitoring
- Anomaly detection, metrics server

**TOTAL IDENTIFIED: 80+ Components**

## Critical Issues Identified

### Issue 1: Equity Calculation Inconsistency
**Problem:** Multiple conflicting equity calculations in the same function
```python
# Line 722: Initial equity from state
current_equity = state.get("current_equity", account_balance)

# Line 1373: Recalculated during simulation
current_equity = account_balance + state["simulated_pnl"]  

# Line 1536: Recalculated again at end
current_equity = account_balance + simulated_pnl
```

### Issue 2: Dashboard Data Not Updating
**Root Cause:** State updates happen AFTER dashboard logging
- Dashboard logs at line 1068 with old `current_equity` 
- Equity updates happen at lines 1536-1571
- State file written at line 1589

### Issue 3: Component Integration Gaps
**Missing from signal generation:**
- Regime forecasting not used in final signal aggregation
- Meta score computed but not used in final decision
- Many technical indicators computed but not feeding into strategies

### Issue 4: Demo Mode Parity Issues
**Problems:**
- Different equity calculation paths for demo vs live
- Inconsistent P&L simulation timing
- Dashboard shows stale data due to update order
