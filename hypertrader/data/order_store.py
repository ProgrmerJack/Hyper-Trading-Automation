import sqlite3
from pathlib import Path
from typing import Iterable, Tuple

class OrderStore:
    """Tiny SQLite-backed store for open orders."""

    def __init__(self, path: str | Path = "orders.db") -> None:
        self.path = Path(path)
        self.conn = sqlite3.connect(self.path)
        self._init_db()

    def _init_db(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS orders (
                id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                volume REAL NOT NULL,
                ts REAL NOT NULL
            )
            """
        )
        self.conn.commit()

    def record(self, order_id: str, symbol: str, side: str, volume: float, ts: float) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO orders(id, symbol, side, volume, ts) VALUES (?,?,?,?,?)",
            (order_id, symbol, side, volume, ts),
        )
        self.conn.commit()

    def remove(self, order_id: str) -> None:
        self.conn.execute("DELETE FROM orders WHERE id=?", (order_id,))
        self.conn.commit()

    def fetch_all(self) -> Iterable[Tuple[str, str, str, float, float]]:
        cur = self.conn.execute("SELECT id, symbol, side, volume, ts FROM orders")
        return cur.fetchall()

    def close(self) -> None:
        self.conn.close()
