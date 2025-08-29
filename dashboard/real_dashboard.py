# Real-time Performance Dashboard for HyperTrader (no simulated P&L)
# Usage:
#   streamlit run dashboard/real_dashboard.py -- --state data/state.json --oms path/to/OMS_DB.sqlite
import argparse, json, sqlite3
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st

def load_state(state_path: Path) -> dict:
    if not state_path.exists():
        return {}
    with state_path.open("r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            return {}

def _read_sql(conn, q, parse_ts=True):
    try:
        return pd.read_sql_query(q, conn, parse_dates=["ts"] if parse_ts else None)
    except Exception:
        df = pd.read_sql_query(q, conn)
        if "ts" in df.columns:
            df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True).dt.tz_convert(None)
        return df

def load_fills(conn) -> pd.DataFrame:
    q = "SELECT id, ts, symbol, side, qty, price, fee, order_id FROM fills ORDER BY ts ASC"
    return _read_sql(conn, q)

def load_positions(conn) -> pd.DataFrame:
    q = "SELECT id, symbol, qty, avg_price, ts FROM positions ORDER BY ts ASC"
    return _read_sql(conn, q)

def realized_pnl_from_fills(fills: pd.DataFrame) -> pd.DataFrame:
    if fills.empty:
        return pd.DataFrame(columns=["ts","symbol","realized_pnl"])
    df = fills.copy()
    df["signed_qty"] = np.where(df["side"].str.upper()=="BUY", df["qty"], -df["qty"])
    df["fee"] = df.get("fee", pd.Series(0.0, index=df.index)).fillna(0.0)
    out = []
    inv = {}
    for row in df.itertuples(index=False):
        sym = row.symbol; signed_q = row.signed_qty; px = row.price; fee = 0.0 if pd.isna(row.fee) else row.fee
        inv.setdefault(sym, [])
        if signed_q > 0:
            inv[sym].append([signed_q, px])
        else:
            q_to_close = -signed_q; pnl = 0.0
            while q_to_close > 0 and inv[sym]:
                lot_qty, lot_px = inv[sym][0]
                close_qty = min(lot_qty, q_to_close)
                pnl += (px - lot_px) * close_qty
                lot_qty -= close_qty; q_to_close -= close_qty
                if lot_qty == 0: inv[sym].pop(0)
                else: inv[sym][0][0] = lot_qty
            out.append({"ts": row.ts, "symbol": sym, "realized_pnl": pnl - fee})
    return pd.DataFrame(out)

def equity_curve(initial_cash: float, realized: pd.DataFrame) -> pd.DataFrame:
    if realized.empty:
        return pd.DataFrame({"ts": [], "equity": []})
    realized = realized.sort_values("ts")
    realized["cum_pnl"] = realized["realized_pnl"].cumsum()
    realized["equity"] = initial_cash + realized["cum_pnl"]
    return realized[["ts","equity"]]

def drawdown_stats(equity_df: pd.DataFrame):
    if equity_df.empty:
        return 0.0, 0.0
    eq = equity_df["equity"].values
    peaks = np.maximum.accumulate(eq)
    dd = (eq - peaks) / peaks
    return float(dd.min()), float(dd[-1])

def var_95(pnl_series: pd.Series) -> float:
    if pnl_series.empty: return 0.0
    return float(np.percentile(pnl_series, 5))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=str, default="data/state.json")
    parser.add_argument("--oms", type=str, required=True, help="Path to OMS SQLite")
    args, _ = parser.parse_known_args()

    st.set_page_config(page_title="HyperTrader – Real Dashboard", layout="wide")
    st.title("📈 HyperTrader — Real Performance Dashboard")

    state_path = Path(args.state); oms_path = Path(args.oms)
    st.sidebar.header("Sources")
    st.sidebar.write(f"State JSON: `{state_path}`")
    st.sidebar.write(f"OMS DB: `{oms_path}`")

    state = load_state(state_path)
    initial_cash = float(state.get("original_balance", state.get("account_balance", 100.0)))

    if not oms_path.exists():
        st.error("OMS SQLite file not found. Pass --oms path/to/OMS_DB.sqlite")
        st.stop()
    conn = sqlite3.connect(str(oms_path))
    with conn:
        fills = load_fills(conn)
        positions = load_positions(conn)

    st.subheader("Orders & Fills")
    st.dataframe(fills.tail(250))

    st.subheader("Open Positions")
    st.dataframe(positions.tail(100))

    st.subheader("Realized P&L & Equity")
    realized = realized_pnl_from_fills(fills)
    st.dataframe(realized.tail(250))

    eq = equity_curve(initial_cash, realized)
    if not eq.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=eq["ts"], y=eq["equity"], mode="lines", name="Equity"))
        fig.update_layout(height=350, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    max_dd, _ = drawdown_stats(eq) if not eq.empty else (0.0, 0.0)
    daily = realized.copy()
    if not daily.empty:
        daily["date"] = pd.to_datetime(daily["ts"]).dt.date
        daily = daily.groupby("date")["realized_pnl"].sum()
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Max Drawdown", f"{max_dd*100:.2f}%")
    with col2: st.metric("VaR (95%, daily)", f"{var_95(daily):.2f}")
    with col3: st.metric("Start Cash", f"${initial_cash:,.2f}")
    with col4:
        latest_eq = float(eq["equity"].iloc[-1]) if not eq.empty else initial_cash
        st.metric("Current Equity", f"${latest_eq:,.2f}")

    st.subheader("Active Components (from state.json)")
    st.json(state.get("active_components", {}))

if __name__ == "__main__":
    main()
