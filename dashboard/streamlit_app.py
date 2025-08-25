from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv


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
    st.title("HyperTrader – Live Dashboard")

    db_default = Path(os.getenv("OMS_DB", "data/state.db"))
    state_default = Path(os.getenv("STATE_JSON", "state.json"))

    with st.sidebar:
        st.header("Controls")
        db_path_str = st.text_input("OMS SQLite path", str(db_default))
        state_path_str = st.text_input("State JSON path", str(state_default))
        symbol_filter = st.text_input("Filter symbol", "")
        symbol_override = st.text_input("Cancel-All symbol (optional)", "")
        if st.button("Cancel All", type="primary"):
            msg = cancel_all_orders(symbol_override or None)
            st.success(msg)
        st.caption("Set LOG_FILE for JSON logs. Metrics at /metrics if enabled.")

    db_path = Path(db_path_str)
    state_path = Path(state_path_str)
    state = load_state(state_path)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Equity", f"{state.get('equity', 0):,.2f}")
    with col2:
        st.metric("Peak Equity", f"{state.get('peak_equity', 0):,.2f}")
    with col3:
        lat = state.get('latencies') or []
        st.metric("Last Latency (s)", f"{(lat[-1] if lat else 0):.3f}")
    with col4:
        st.metric("Latency breaches", str(state.get("latency_breach", 0)))

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

    # Charts
    st.subheader("Equity & Latency (Charts)")
    ch1, ch2 = st.columns(2)
    with ch1:
        eq_hist = state.get("equity_history", [])
        if eq_hist:
            try:
                eq_df = pd.DataFrame(eq_hist, columns=["ts", "equity"]).assign(ts=lambda d: pd.to_datetime(d.ts))
                eq_df.set_index("ts", inplace=True)
                st.line_chart(eq_df["equity"], use_container_width=True)
            except Exception:
                st.info("No equity history yet.")
        else:
            st.info("No equity history yet.")
    with ch2:
        lat = state.get("latencies") or []
        if lat:
            lat_df = pd.DataFrame({"latency": pd.Series(lat)}).astype(float)
            st.bar_chart(lat_df.tail(100), use_container_width=True)
        else:
            st.info("No latency samples yet.")

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


