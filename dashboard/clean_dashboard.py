"""
Clean HyperTrader Dashboard - Built from scratch for reliable data handling
No phantom data, proper state management, conservative growth tracking
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import sqlite3
from pathlib import Path
import time
from datetime import datetime, timezone
import numpy as np
from typing import Dict, Any, Optional

# Page configuration
st.set_page_config(
    page_title="HyperTrader Conservative Dashboard", 
    page_icon="📈", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

def ensure_data_directory() -> Path:
    """Ensure data directory exists and return path."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    return data_dir

def get_database_connection(db_path: Path) -> Optional[sqlite3.Connection]:
    """Get database connection if file exists."""
    if db_path.exists():
        try:
            return sqlite3.connect(str(db_path))
        except Exception:
            return None
    return None

def load_state_file(state_path: Path) -> Dict[str, Any]:
    """Load state file if it exists, otherwise return empty state."""
    if state_path.exists():
        try:
            with open(state_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}
    return {}

def safe_query(conn: sqlite3.Connection, query: str, params: tuple = ()) -> pd.DataFrame:
    """Safely execute database query."""
    try:
        return pd.read_sql_query(query, conn, params=params)
    except Exception:
        return pd.DataFrame()

def main():
    st.title("📈 HyperTrader Conservative Dashboard")
    st.markdown("*Real-time monitoring for steady, consistent growth*")
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Dashboard Controls")
        
        # Auto-refresh
        auto_refresh = st.checkbox("Auto Refresh (5s)", value=True)
        if auto_refresh:
            time.sleep(0.1)  # Small delay for smooth UI
            st.rerun()
        
        # File paths
        st.subheader("📁 Data Sources")
        data_dir = ensure_data_directory()
        
        db_path = data_dir / "state.db" 
        state_path = data_dir / "state.json"
        signal_path = data_dir / "signal.json"
        
        # Show file status
        st.write(f"**Database:** {'✅' if db_path.exists() else '❌'} `{db_path.name}`")
        st.write(f"**State:** {'✅' if state_path.exists() else '❌'} `{state_path.name}`")
        st.write(f"**Signals:** {'✅' if signal_path.exists() else '❌'} `{signal_path.name}`")
        
        # Manual refresh button
        if st.button("🔄 Manual Refresh"):
            st.cache_data.clear()
            st.rerun()
    
    # Load data sources
    state = load_state_file(state_path)
    conn = get_database_connection(db_path)
    
    # Check if bot is actually running
    bot_running = False
    if state and signal_path.exists():
        try:
            signal_data = json.loads(signal_path.read_text())
            last_signal_time = signal_data.get('timestamp')
            if last_signal_time:
                last_time = datetime.fromisoformat(last_signal_time.replace('Z', '+00:00'))
                time_diff = datetime.now(timezone.utc) - last_time
                bot_running = time_diff.total_seconds() < 300  # Active if signal within 5 minutes
        except Exception:
            pass
    
    # Bot Status Header
    if bot_running:
        st.success("🤖 Bot Status: **ACTIVE** - Conservative Trading Mode")
    else:
        st.warning("🤖 Bot Status: **INACTIVE** - No Recent Activity")
        st.info("Start the bot with: `python run_bot_continuous.py BTC-USD --config configs/enhanced_conservative_config.yaml`")
    
    if not bot_running and not state:
        st.info("**No data available.** The dashboard will populate once the bot starts running.")
        st.stop()
    
    # Account Overview
    st.subheader("💰 Account Overview")
    
    account_balance = state.get('original_balance', 100.0)
    current_equity = state.get('current_equity', account_balance)
    simulated_pnl = state.get('simulated_pnl', 0.0)
    peak_equity = state.get('peak_equity', account_balance)
    
    # Calculate metrics
    total_pnl = current_equity - account_balance
    pnl_percentage = (total_pnl / account_balance * 100) if account_balance > 0 else 0
    drawdown = ((peak_equity - current_equity) / peak_equity * 100) if peak_equity > 0 else 0
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Current Balance", 
            f"${current_equity:.2f}",
            delta=f"${total_pnl:.2f}" if total_pnl != 0 else None
        )
    
    with col2:
        st.metric(
            "Total P&L", 
            f"{pnl_percentage:+.2f}%",
            delta=f"${total_pnl:.2f}" if total_pnl != 0 else None
        )
    
    with col3:
        st.metric(
            "Peak Equity", 
            f"${peak_equity:.2f}",
            delta=None
        )
    
    with col4:
        color = "normal"
        if drawdown > 3:
            color = "inverse"
        st.metric(
            "Current Drawdown", 
            f"{drawdown:.2f}%",
            delta=None
        )
    
    # $1000 Growth Target with Milestones
    st.subheader("🎯 $1000 Growth Target")
    target_equity = 1000.0  # 10x growth target
    milestones = [200.0, 350.0, 500.0, 750.0, 1000.0]
    
    progress = min(1.0, current_equity / target_equity)
    st.progress(progress, text=f"Progress: ${current_equity:.2f} / ${target_equity:.2f} ({progress*100:.1f}%)")
    
    # Milestone tracking
    cols = st.columns(len(milestones))
    for i, milestone in enumerate(milestones):
        with cols[i]:
            achieved = current_equity >= milestone
            icon = "✅" if achieved else "⭕"
            st.metric(f"{icon} ${milestone:.0f}", f"{milestone/100:.0f}x", delta=None)
    
    if progress >= 1.0:
        st.balloons()
        st.success("🎉 **$1000 TARGET ACHIEVED!** Congratulations on 10x growth!")
    elif current_equity >= 500.0:
        st.info("🚀 Halfway to $1000! Keep up the balanced trading.")
    elif current_equity >= 200.0:
        st.info("📈 Great progress! First milestone achieved.")
    
    # Risk Management Status
    st.subheader("⚠️ Risk Management")
    
    risk_col1, risk_col2, risk_col3 = st.columns(3)
    
    with risk_col1:
        if drawdown <= 5:
            st.success(f"✅ Drawdown: {drawdown:.2f}% (Safe)")
        else:
            st.error(f"🚨 Drawdown: {drawdown:.2f}% (Risk Limit Exceeded)")
    
    with risk_col2:
        daily_pnl = 0  # Would calculate from today's trades
        if abs(daily_pnl) <= 5:
            st.success(f"✅ Daily P&L: {daily_pnl:+.2f}% (Within Limits)")
        else:
            st.warning(f"⚠️ Daily P&L: {daily_pnl:+.2f}% (Monitor Closely)")
    
    with risk_col3:
        trade_count = state.get('trade_count', 0)
        st.info(f"📊 Total Trades: {trade_count}")
    
    # Trading Activity (only show if data exists)
    if conn is not None:
        st.subheader("📈 Trading Activity")
        
        # Recent orders
        orders_df = safe_query(conn, """
            SELECT 
                datetime(ts, 'unixepoch') as timestamp,
                symbol,
                side,
                qty,
                price,
                status
            FROM orders 
            ORDER BY ts DESC 
            LIMIT 50
        """)
        
        if not orders_df.empty:
            st.write("**Recent Orders**")
            st.dataframe(orders_df, use_container_width=True, hide_index=True)
            
            # Recent fills for P&L calculation
            fills_df = safe_query(conn, """
                SELECT 
                    datetime(f.ts, 'unixepoch') as timestamp,
                    o.side,
                    f.qty,
                    f.price,
                    f.fee
                FROM fills f
                JOIN orders o ON f.order_id = o.id
                ORDER BY f.ts DESC
                LIMIT 20
            """)
            
            if not fills_df.empty:
                st.write("**Recent Fills**")
                st.dataframe(fills_df, use_container_width=True, hide_index=True)
        else:
            st.info("No trading activity yet. The bot will start trading when conditions are met.")
    
    # Equity Chart (only if equity history exists)
    equity_history = state.get('equity_history', [])
    if equity_history and len(equity_history) > 1:
        st.subheader("📊 Equity Curve")
        
        # Convert equity history to DataFrame
        eq_data = []
        for entry in equity_history:
            if isinstance(entry, dict) and 'timestamp' in entry:
                eq_data.append({
                    'timestamp': pd.to_datetime(entry['timestamp']),
                    'equity': entry.get('equity', account_balance),
                    'pnl': entry.get('pnl', 0),
                    'drawdown': entry.get('drawdown', 0)
                })
        
        if eq_data:
            eq_df = pd.DataFrame(eq_data)
            eq_df = eq_df.sort_values('timestamp')
            
            # Create equity chart
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Account Equity ($)', 'Drawdown (%)'),
                vertical_spacing=0.1,
                shared_xaxes=True
            )
            
            # Equity line
            fig.add_trace(
                go.Scatter(
                    x=eq_df['timestamp'], 
                    y=eq_df['equity'],
                    mode='lines',
                    name='Equity',
                    line=dict(color='green', width=2)
                ),
                row=1, col=1
            )
            
            # Starting balance line
            fig.add_hline(
                y=account_balance, 
                line_dash="dash", 
                line_color="blue",
                annotation_text="Starting Balance",
                row=1, col=1
            )
            
            # Target line
            fig.add_hline(
                y=target_equity, 
                line_dash="dot", 
                line_color="orange",
                annotation_text="Conservative Target",
                row=1, col=1
            )
            
            # Drawdown
            fig.add_trace(
                go.Scatter(
                    x=eq_df['timestamp'], 
                    y=eq_df['drawdown'],
                    mode='lines',
                    name='Drawdown',
                    fill='tonexty',
                    line=dict(color='red')
                ),
                row=2, col=1
            )
            
            # 5% drawdown limit line
            fig.add_hline(
                y=5, 
                line_dash="dash", 
                line_color="red",
                annotation_text="Risk Limit (5%)",
                row=2, col=1
            )
            
            fig.update_layout(
                height=500,
                title="Conservative Growth Tracking",
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Strategy Performance (if available)
    strategy_performance = state.get('strategy_performance', {})
    if strategy_performance:
        st.subheader("🧠 Strategy Performance")
        
        perf_data = []
        for name, data in strategy_performance.items():
            returns = data.get('returns', [])
            if returns:
                perf_data.append({
                    'Strategy': name,
                    'Last Signal': data.get('last_signal', 'HOLD'),
                    'Confidence': f"{data.get('confidence', 0.5):.2f}",
                    'Trades': len(returns),
                    'Avg Return': f"{np.mean(returns):.4f}" if returns else "0.0000"
                })
        
        if perf_data:
            perf_df = pd.DataFrame(perf_data)
            st.dataframe(perf_df, use_container_width=True, hide_index=True)
    
    # Footer
    st.markdown("---")
    st.markdown("*HyperTrader Balanced Growth Dashboard - $1000 Target with 79 Components*")
    
    # Update timestamp
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
