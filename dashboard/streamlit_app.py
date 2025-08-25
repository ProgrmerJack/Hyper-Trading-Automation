from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import pytz
from datetime import datetime


load_dotenv()


def load_state(state_path: Path) -> dict:
    if not state_path.exists():
        return {}
    try:
        return json.loads(state_path.read_text())
    except Exception:
        return {}


def query_df(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> pd.DataFrame:
    try:
        return pd.read_sql_query(sql, conn, params=params)
    except Exception:
        return pd.DataFrame()


def cancel_all_orders(symbol: str | None = None) -> str:
    try:
        from hypertrader.execution.ccxt_executor import cancel_all

        asyncio.run(cancel_all(symbol))
        return "Cancel-all sent"
    except Exception as e:
        return f"Cancel-all failed: {e}"


def main() -> None:
    st.set_page_config(page_title="HyperTrader Dashboard", layout="wide")
    st.title("🚀 HyperTrader – Live Real-Time Dashboard")
    
    # Uzbekistan timezone display
    try:
        uzbekistan_tz = pytz.timezone('Asia/Tashkent')
        utc_now = datetime.utcnow().replace(tzinfo=pytz.UTC)
        current_time_uzb = utc_now.astimezone(uzbekistan_tz)
        st.caption(f"🇺🇿 Uzbekistan Time: {current_time_uzb.strftime('%Y-%m-%d %H:%M:%S %Z')} (UTC+5)")
    except Exception as e:
        st.caption(f"🇺🇿 Timezone Error: {e}")
    
    # Auto-refresh every 5 seconds for real-time data
    import time
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = time.time()
    
    # Auto-refresh mechanism
    refresh_placeholder = st.empty()
    with refresh_placeholder.container():
        st.info("🔄 Dashboard auto-refreshes every 5 seconds for real-time data")
    
    # Auto-refresh using Streamlit's rerun capability
    if time.time() - st.session_state.last_refresh > 5:
        st.session_state.last_refresh = time.time()
        st.rerun()

    db_default = Path(os.getenv("OMS_DB", "data/state.db"))
    state_default = Path(os.getenv("STATE_JSON", "data/state.json"))
    
    # Load state first
    state_path = Path(str(state_default))
    state = load_state(state_path)

    with st.sidebar:
        st.header("🎛️ Controls")
        advanced = st.checkbox("Advanced paths", value=False)
        
        # Component Status
        st.header("🔧 System Status")
        components = state.get("active_components", {})
        if components:
            st.success(f"✅ {components.get('total_active', 0)} Strategies Active")
            st.success(f"✅ {len(components.get('indicators', []))} Indicators Active")
            if components.get('macro_sentiment'):
                st.success("✅ Macro Sentiment Analysis")
            else:
                st.warning("⚠️ Macro Sentiment Inactive")
            if components.get('micro_sentiment'):
                st.success("✅ Microstructure Analysis")
            else:
                st.warning("⚠️ Microstructure Inactive")
            if components.get('ml_models'):
                st.success("✅ ML Models Active")
            else:
                st.info("ℹ️ ML Models Pending")
            
            st.info(f"📊 Position Sizing: {components.get('position_sizing', 'Standard')}")
        else:
            st.error("❌ No component data")
        if advanced:
            db_path_str = st.text_input("OMS SQLite path", str(db_default))
            state_path_str = st.text_input("State JSON path", str(state_default))
        else:
            db_path_str = str(db_default)
            state_path_str = str(state_default)
        symbol_filter = st.text_input("Filter symbol", "")
        symbol_override = st.text_input("Cancel-All symbol (optional)", "")
        if st.button("Cancel All", type="primary"):
            msg = cancel_all_orders(symbol_override or None)
            st.success(msg)
        st.caption("Set LOG_FILE for JSON logs. Metrics at /metrics if enabled.")

    db_path = Path(db_path_str)
    if advanced:
        state_path = Path(state_path_str)
        state = load_state(state_path)

    # Calculate P&L metrics for $100 to $1000 challenge
    current_equity = state.get('equity', 0)
    initial_equity = state.get('original_balance', 100)  # Use original starting balance for P&L
    total_pnl = current_equity - initial_equity
    pnl_pct = (total_pnl / initial_equity * 100) if initial_equity > 0 else 0
    peak_equity = state.get('peak_equity', 0)
    drawdown = ((peak_equity - current_equity) / peak_equity * 100) if peak_equity > 0 else 0

    # 10x Challenge Progress Bar
    target_equity = 1000.0
    progress = min(1.0, current_equity / target_equity)
    st.subheader("🎯 $100 → $1000 Challenge Progress")
    st.progress(progress, text=f"${current_equity:.2f} / $1000 ({progress*100:.1f}%)")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Current Equity", f"${current_equity:,.2f}")
    with col2:
        st.metric("Total P&L", f"${total_pnl:,.2f}", f"{pnl_pct:+.2f}%")
    with col3:
        remaining = target_equity - current_equity
        st.metric("To Target", f"${remaining:,.2f}", f"{((target_equity/current_equity-1)*100) if current_equity > 0 else 0:.1f}% more")
    with col4:
        st.metric("Drawdown", f"{drawdown:.2f}%")
    with col5:
        lat = state.get('latencies') or []
        st.metric("Last Latency", f"{(lat[-1] if lat else 0):.3f}s")

    if not db_path.exists():
        st.warning(f"OMS database not found: {db_path}")
        st.stop()

    conn = sqlite3.connect(db_path)
    try:
        base_where = "" if not symbol_filter else " WHERE symbol = ? "
        params = tuple() if not symbol_filter else (symbol_filter,)
        open_orders = query_df(
            conn,
            (
                "SELECT id, client_id, symbol, side, qty, price, status, "
                "datetime(ts, 'unixepoch') AS ts FROM orders "
                f"{base_where}AND status NOT IN ('FILLED','CANCELED') ORDER BY ts DESC"
                if base_where
                else "SELECT id, client_id, symbol, side, qty, price, status, datetime(ts, 'unixepoch') AS ts FROM orders WHERE status NOT IN ('FILLED','CANCELED') ORDER BY ts DESC"
            ),
            params,
        )
        recent_orders = query_df(
            conn,
            (
                "SELECT id, client_id, symbol, side, qty, price, status, "
                "datetime(ts, 'unixepoch') AS ts FROM orders "
                f"{base_where}ORDER BY ts DESC LIMIT 200"
            ),
            params,
        )
        fills = query_df(
            conn,
            (
                "SELECT order_id, qty, price, fee, datetime(ts, 'unixepoch') AS ts FROM fills "
                f"{'WHERE order_id IN (SELECT id FROM orders WHERE symbol = ?)' if symbol_filter else ''} "
                "ORDER BY ts DESC LIMIT 200"
            ),
            params,
        )
        positions = query_df(
            conn,
            (
                "SELECT symbol, qty, entry_px, liq_px, datetime(ts, 'unixepoch') AS ts FROM positions "
                f"{base_where}ORDER BY ts DESC"
            ),
            params,
        )
    finally:
        conn.close()

    st.subheader("Open Orders")
    st.dataframe(open_orders, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Recent Orders")
        st.dataframe(recent_orders, use_container_width=True, height=400, hide_index=True)
        if not recent_orders.empty:
            csv = recent_orders.to_csv(index=False).encode()
            st.download_button("Download Orders CSV", csv, file_name="orders.csv", mime="text/csv")
    with c2:
        st.subheader("Recent Fills")
        st.dataframe(fills, use_container_width=True, height=400, hide_index=True)

    st.subheader("Positions")
    st.dataframe(positions, use_container_width=True, hide_index=True)

    st.caption("Tip: use the filter to focus on a symbol. Click browser refresh to update.")

    # Advanced Performance Metrics
    st.subheader("📊 Advanced Performance & Risk Analysis")
    
    # Real-time trading metrics
    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
    with perf_col1:
        # Calculate Sharpe ratio from equity history
        eq_hist = state.get("equity_history", [])
        if len(eq_hist) > 10:
            eq_df = pd.DataFrame(eq_hist, columns=["ts", "equity"])
            eq_df["returns"] = eq_df["equity"].pct_change().dropna()
            sharpe = (eq_df["returns"].mean() / eq_df["returns"].std() * (365**0.5)) if eq_df["returns"].std() > 0 else 0
        else:
            sharpe = 0
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
    
    with perf_col2:
        # Maximum consecutive wins/losses
        perf = state.get("strategy_performance", {})
        total_trades = sum(len(s.get("returns", [])) for s in perf.values())
        st.metric("Total Trades", f"{total_trades}")
    
    with perf_col3:
        # Calculate profit factor
        if eq_hist and len(eq_hist) > 1:
            gross_profit = sum(max(0, eq_hist[i][1] - eq_hist[i-1][1]) for i in range(1, len(eq_hist)))
            gross_loss = sum(min(0, eq_hist[i][1] - eq_hist[i-1][1]) for i in range(1, len(eq_hist)))
            profit_factor = abs(gross_profit / gross_loss) if gross_loss < 0 else float('inf')
            st.metric("Profit Factor", f"{profit_factor:.2f}")
        else:
            st.metric("Profit Factor", "N/A")
    
    with perf_col4:
        # Max consecutive trades
        lat_breach = state.get("latency_breach", 0)
        st.metric("Latency Breaches", f"{lat_breach}")

    # Detailed P&L Analysis
    st.subheader("💰 Detailed Profit & Loss Analysis")
    pnl_col1, pnl_col2 = st.columns(2)
    
    with pnl_col1:
        conn = sqlite3.connect(db_path)
        try:
            # Calculate realized P&L from fills
            fills_pnl = query_df(conn, """
                SELECT 
                    o.symbol,
                    o.side,
                    f.qty,
                    f.price,
                    f.fee,
                    f.ts,
                    (CASE WHEN o.side = 'SELL' THEN f.qty * f.price 
                          ELSE -f.qty * f.price END) as realized_pnl
                FROM fills f
                JOIN orders o ON f.order_id = o.id
                ORDER BY f.ts DESC
                LIMIT 100
            """)
            
            if not fills_pnl.empty:
                fills_pnl["cumulative_pnl"] = fills_pnl["realized_pnl"].cumsum()
                fills_pnl["ts"] = pd.to_datetime(fills_pnl["ts"], unit="s")
                
                st.write("**Realized P&L Over Time**")
                st.line_chart(fills_pnl.set_index("ts")["cumulative_pnl"], use_container_width=True)
                
                # P&L summary
                total_realized = fills_pnl["realized_pnl"].sum()
                total_fees = fills_pnl["fee"].sum()
                win_trades = len(fills_pnl[fills_pnl["realized_pnl"] > 0])
                total_trades = len(fills_pnl)
                win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0
                
                st.metric("Realized P&L", f"${total_realized:,.2f}")
                st.metric("Total Fees", f"${total_fees:,.2f}")
                st.metric("Win Rate", f"{win_rate:.1f}%")
            else:
                st.info("No trade data for P&L analysis yet.")
        finally:
            conn.close()
    
    with pnl_col2:
        # Unrealized P&L from positions
        conn = sqlite3.connect(db_path)
        try:
            current_positions = query_df(conn, "SELECT symbol, qty, entry_px FROM positions WHERE qty != 0")
            if not current_positions.empty:
                st.write("**Current Positions & Unrealized P&L**")
                # For demo, assume current price is close to recent fills
                recent_price_query = query_df(conn, """
                    SELECT AVG(price) as current_price 
                    FROM fills 
                    WHERE ts > (SELECT MAX(ts) - 3600 FROM fills)
                """)
                current_price = recent_price_query.iloc[0]["current_price"] if not recent_price_query.empty else 0
                
                pos_data = []
                total_unrealized = 0
                for _, pos in current_positions.iterrows():
                    if current_price > 0:
                        unrealized = pos["qty"] * (current_price - pos["entry_px"])
                        total_unrealized += unrealized
                    else:
                        unrealized = 0
                    pos_data.append({
                        "Symbol": pos["symbol"],
                        "Qty": f"{pos['qty']:.4f}",
                        "Entry Price": f"${pos['entry_px']:.2f}",
                        "Current Price": f"${current_price:.2f}",
                        "Unrealized P&L": f"${unrealized:.2f}"
                    })
                
                st.dataframe(pd.DataFrame(pos_data), use_container_width=True, hide_index=True)
                st.metric("Total Unrealized P&L", f"${total_unrealized:.2f}")
            else:
                st.info("No open positions.")
        finally:
            conn.close()

    # Advanced Multi-Dimensional Charts
    st.subheader("📈 Advanced Performance Visualization")
    
    # Multi-chart layout with sophisticated analysis
    chart_tabs = st.tabs(["📊 Equity Analysis", "⚡ Performance Metrics", "🎯 Strategy Breakdown", "📋 Risk Metrics"])
    
    with chart_tabs[0]:
        ch1, ch2 = st.columns(2)
        with ch1:
            eq_hist = state.get("equity_history", [])
            if eq_hist and len(eq_hist) > 2:
                try:
                    eq_df = pd.DataFrame(eq_hist, columns=["ts", "equity"]).assign(ts=lambda d: pd.to_datetime(d.ts))
                    eq_df.set_index("ts", inplace=True)
                    eq_df["pnl_from_start"] = eq_df["equity"] - initial_equity
                    eq_df["pnl_pct"] = (eq_df["equity"] / initial_equity - 1) * 100
                    eq_df["rolling_max"] = eq_df["equity"].expanding().max()
                    eq_df["drawdown"] = (eq_df["equity"] / eq_df["rolling_max"] - 1) * 100
                    
                    st.write("**📈 Equity Curve with Drawdown**")
                    
                    # Create sophisticated equity chart
                    import plotly.graph_objects as go
                    from plotly.subplots import make_subplots
                    
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                      subplot_titles=('Equity ($)', 'Drawdown (%)'),
                                      vertical_spacing=0.1)
                    
                    # Equity line
                    fig.add_trace(go.Scatter(x=eq_df.index, y=eq_df["equity"], 
                                           mode='lines', name='Equity', 
                                           line=dict(color='green', width=2)), row=1, col=1)
                    
                    # Peak equity line
                    fig.add_trace(go.Scatter(x=eq_df.index, y=eq_df["rolling_max"],
                                           mode='lines', name='Peak Equity',
                                           line=dict(color='blue', dash='dash')), row=1, col=1)
                    
                    # Drawdown
                    fig.add_trace(go.Scatter(x=eq_df.index, y=eq_df["drawdown"],
                                           mode='lines', name='Drawdown',
                                           fill='tonexty', fillcolor='rgba(255,0,0,0.3)',
                                           line=dict(color='red')), row=2, col=1)
                    
                    fig.update_layout(height=500, showlegend=True,
                                    title="Real-Time Equity Performance")
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Chart error: {e}")
                    st.line_chart(eq_df["equity"], use_container_width=True)
            else:
                st.info("Accumulating equity history...")
        
        with ch2:
            # Volume and trade frequency analysis
            conn = sqlite3.connect(db_path)
            try:
                volume_data = query_df(conn, """
                    SELECT 
                        DATE(datetime(ts, 'unixepoch')) as date,
                        COUNT(*) as trade_count,
                        SUM(qty * price) as volume,
                        AVG(price) as avg_price
                    FROM fills f
                    JOIN orders o ON f.order_id = o.id
                    GROUP BY DATE(datetime(ts, 'unixepoch'))
                    ORDER BY date DESC
                    LIMIT 30
                """)
                
                if not volume_data.empty:
                    st.write("**📊 Daily Trading Volume & Frequency**")
                    
                    import plotly.express as px
                    fig = px.bar(volume_data.tail(10), x='date', y='volume',
                               hover_data=['trade_count', 'avg_price'],
                               title="Daily Trading Volume")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No volume data yet.")
            finally:
                conn.close()
    
    with chart_tabs[1]:
        perf_ch1, perf_ch2 = st.columns(2)
        with perf_ch1:
            # Returns distribution
            if eq_hist and len(eq_hist) > 10:
                eq_df = pd.DataFrame(eq_hist, columns=["ts", "equity"])
                eq_df["returns"] = eq_df["equity"].pct_change().dropna() * 100
                
                st.write("**📊 Returns Distribution**")
                try:
                    import plotly.express as px
                    fig = px.histogram(eq_df, x="returns", nbins=20, 
                                     title="Returns Distribution (%)")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    # Fallback to simple histogram
                    st.bar_chart(eq_df["returns"].hist(bins=20))
        
        with perf_ch2:
            lat = state.get("latencies") or []
            if lat:
                st.write("**⚡ Latency Performance**")
                lat_df = pd.DataFrame({"latency": pd.Series(lat)}).astype(float)
                
                import plotly.express as px
                fig = px.histogram(lat_df.tail(100), x="latency", nbins=20,
                                 title="Latency Distribution (ms)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No latency data yet.")
    
    with chart_tabs[2]:
        # Strategy performance breakdown
        perf = state.get("strategy_performance", {})
        if perf:
            st.write("**🎯 Individual Strategy Performance**")
            
            strategy_data = []
            for name, data in perf.items():
                returns = data.get("returns", [])
                if returns:
                    total_return = sum(returns)
                    win_rate = len([r for r in returns if r > 0]) / len(returns) * 100
                    avg_return = sum(returns) / len(returns)
                    strategy_data.append({
                        "Strategy": name,
                        "Total Return": total_return,
                        "Win Rate (%)": win_rate,
                        "Avg Return": avg_return,
                        "Trade Count": len(returns),
                        "Last Signal": data.get("last_signal", "HOLD"),
                        "Confidence": data.get("confidence", 0.5)
                    })
            
            if strategy_data:
                strategy_df = pd.DataFrame(strategy_data)
                
                # Strategy performance chart
                import plotly.express as px
                fig = px.scatter(strategy_df, x="Win Rate (%)", y="Total Return",
                               size="Trade Count", color="Confidence",
                               hover_name="Strategy",
                               title="Strategy Performance Matrix")
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed strategy table
                st.dataframe(strategy_df.sort_values("Total Return", ascending=False),
                           use_container_width=True, hide_index=True)
        else:
            st.info("No strategy performance data yet.")
    
    with chart_tabs[3]:
        # Risk metrics and analysis
        st.write("**⚠️ Risk Analysis Dashboard**")
        
        risk_col1, risk_col2 = st.columns(2)
        with risk_col1:
            if eq_hist and len(eq_hist) > 5:
                eq_df = pd.DataFrame(eq_hist, columns=["ts", "equity"])
                eq_df["returns"] = eq_df["equity"].pct_change().dropna()
                
                # VaR calculation
                var_95 = eq_df["returns"].quantile(0.05) * 100
                var_99 = eq_df["returns"].quantile(0.01) * 100
                
                st.metric("Value at Risk (95%)", f"{var_95:.2f}%")
                st.metric("Value at Risk (99%)", f"{var_99:.2f}%")
                
                # Maximum drawdown
                eq_df["rolling_max"] = eq_df["equity"].expanding().max()
                eq_df["drawdown"] = (eq_df["equity"] / eq_df["rolling_max"] - 1) * 100
                max_dd = eq_df["drawdown"].min()
                st.metric("Maximum Drawdown", f"{max_dd:.2f}%")
        
        with risk_col2:
            # Position risk breakdown
            conn = sqlite3.connect(db_path)
            try:
                position_risk = query_df(conn, """
                    SELECT 
                        symbol,
                        ABS(qty) * entry_px as exposure,
                        qty,
                        entry_px
                    FROM positions 
                    WHERE qty != 0
                """)
                
                if not position_risk.empty:
                    total_exposure = position_risk["exposure"].sum()
                    st.metric("Total Exposure", f"${total_exposure:,.2f}")
                    
                    # Position exposure chart
                    import plotly.express as px
                    fig = px.pie(position_risk, values="exposure", names="symbol",
                               title="Position Exposure Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No position risk data.")
            finally:
                conn.close()

    # Entry chart: show recent fills over price
    st.subheader("Entries (Last 200)")
    try:
        conn = sqlite3.connect(db_path)
        fills_short = query_df(
            conn,
            "SELECT order_id, qty, price, ts FROM fills ORDER BY ts DESC LIMIT 200",
        )
        conn.close()
        if not fills_short.empty:
            fills_short["ts"] = pd.to_datetime(fills_short["ts"], unit="s")
            fills_short.sort_values("ts", inplace=True)
            st.line_chart(fills_short.set_index("ts")["price"], use_container_width=True)
        else:
            st.info("No fills yet.")
    except Exception:
        st.info("No fills yet.")

    # Strategy votes (if present)
    st.subheader("Strategy Signals (Last)")
    try:
        perf = state.get("strategy_performance", {})
        rows = []
        for name, d in perf.items():
            rows.append({
                "strategy": name,
                "last_signal": d.get("last_signal", "HOLD"),
                "confidence": d.get("confidence", 0.5),
                "recent_return": (d.get("returns", []) or [0])[-1],
            })
        if rows:
            st.dataframe(pd.DataFrame(rows).sort_values(by="confidence", ascending=False), use_container_width=True, hide_index=True)
        else:
            st.info("No strategy performance data yet.")
    except Exception:
        st.info("No strategy performance data yet.")


if __name__ == "__main__":
    main()


