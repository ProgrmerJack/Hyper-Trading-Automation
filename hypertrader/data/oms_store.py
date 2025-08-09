import sqlite3
from pathlib import Path
from typing import Iterable, Tuple


class OMSStore:
    """Tiny SQLite-backed store for orders, fills and positions."""

    def __init__(self, path: str | Path = "oms.db") -> None:
        self.path = Path(path)
        self.conn = sqlite3.connect(self.path)
        self._init_db()

    def _init_db(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS orders (
                id TEXT PRIMARY KEY,
                client_id TEXT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                qty REAL NOT NULL,
                price REAL,
                status TEXT NOT NULL,
                ts REAL NOT NULL
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT NOT NULL,
                qty REAL NOT NULL,
                price REAL NOT NULL,
                fee REAL DEFAULT 0,
                ts REAL NOT NULL
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS positions (
                symbol TEXT PRIMARY KEY,
                qty REAL NOT NULL,
                entry_px REAL,
                liq_px REAL,
                ts REAL NOT NULL
            )
            """
        )
        self.conn.commit()

    # order helpers -----------------------------------------------------
    def record_order(
        self,
        order_id: str,
        client_id: str | None,
        symbol: str,
        side: str,
        qty: float,
        price: float | None,
        status: str,
        ts: float,
    ) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO orders(id, client_id, symbol, side, qty, price, status, ts) VALUES (?,?,?,?,?,?,?,?)",
            (order_id, client_id, symbol, side, qty, price, status, ts),
        )
        self.conn.commit()

    def update_order_status(self, order_id: str, status: str) -> None:
        self.conn.execute("UPDATE orders SET status=? WHERE id=?", (status, order_id))
        self.conn.commit()

    def remove_order(self, order_id: str) -> None:
        self.conn.execute("DELETE FROM orders WHERE id=?", (order_id,))
        self.conn.commit()

    def fetch_open_orders(self) -> Iterable[Tuple[str, str, str, float, float]]:
        cur = self.conn.execute(
            "SELECT id, symbol, side, qty, ts FROM orders WHERE status NOT IN ('FILLED','CANCELED')"
        )
        return cur.fetchall()

    # fills -------------------------------------------------------------
    def record_fill(self, order_id: str, qty: float, price: float, fee: float, ts: float) -> None:
        self.conn.execute(
            "INSERT INTO fills(order_id, qty, price, fee, ts) VALUES (?,?,?,?,?)",
            (order_id, qty, price, fee, ts),
        )
        self.conn.commit()

    # positions ---------------------------------------------------------
    def upsert_position(
        self, symbol: str, qty: float, entry_px: float | None, liq_px: float | None, ts: float
    ) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO positions(symbol, qty, entry_px, liq_px, ts) VALUES (?,?,?,?,?)",
            (symbol, qty, entry_px, liq_px, ts),
        )
        self.conn.commit()

    def fetch_positions(self) -> Iterable[Tuple[str, float, float | None, float | None, float]]:
        cur = self.conn.execute(
            "SELECT symbol, qty, entry_px, liq_px, ts FROM positions"
        )
        return cur.fetchall()

    def close(self) -> None:
        self.conn.close()
