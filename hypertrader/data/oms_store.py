import asyncio
import sqlite3
from pathlib import Path
from typing import Iterable, Tuple


class OMSStore:
    """Tiny SQLite-backed store for orders, fills and positions.

    The store serializes writes through an ``asyncio.Queue`` to avoid the
    ``database is locked`` foot-gun under concurrent access.  Reads are executed
    in a thread via :func:`asyncio.to_thread` to keep the event loop responsive.
    """

    def __init__(self, path: str | Path = "oms.db") -> None:
        self.path = Path(path)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._init_db()
        self._queue: asyncio.Queue[tuple[str | None, tuple]] = asyncio.Queue()
        self._writer = asyncio.create_task(self._writer_loop())

    async def _writer_loop(self) -> None:
        while True:
            sql, params = await self._queue.get()
            if sql is None:
                self._queue.task_done()
                break
            await asyncio.to_thread(self.conn.execute, sql, params)
            await asyncio.to_thread(self.conn.commit)
            self._queue.task_done()

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
                ts REAL NOT NULL,
                UNIQUE(order_id, ts, qty, price, fee)
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

    async def _enqueue(self, sql: str, params: tuple) -> None:
        await self._queue.put((sql, params))
        await self._queue.join()

    # order helpers -----------------------------------------------------
    async def record_order(
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
        await self._enqueue(
            "INSERT OR REPLACE INTO orders(id, client_id, symbol, side, qty, price, status, ts) VALUES (?,?,?,?,?,?,?,?)",
            (order_id, client_id, symbol, side, qty, price, status, ts),
        )

    async def update_order_status(self, order_id: str, status: str) -> None:
        await self._enqueue(
            "UPDATE orders SET status=? WHERE id=?", (status, order_id)
        )

    async def remove_order(self, order_id: str) -> None:
        await self._enqueue("DELETE FROM orders WHERE id=?", (order_id,))

    async def fetch_open_orders(self) -> Iterable[Tuple[str, str, str, float, float]]:
        cur = await asyncio.to_thread(
            self.conn.execute,
            "SELECT id, symbol, side, qty, ts FROM orders WHERE status NOT IN ('FILLED','CANCELED')",
        )
        rows = await asyncio.to_thread(cur.fetchall)
        return rows

    # fills -------------------------------------------------------------
    async def record_fill(
        self, order_id: str, qty: float, price: float, fee: float, ts: float
    ) -> None:
        await self._enqueue(
            "INSERT OR IGNORE INTO fills(order_id, qty, price, fee, ts) VALUES (?,?,?,?,?)",
            (order_id, qty, price, fee, ts),
        )

    # positions ---------------------------------------------------------
    async def upsert_position(
        self, symbol: str, qty: float, entry_px: float | None, liq_px: float | None, ts: float
    ) -> None:
        await self._enqueue(
            "INSERT OR REPLACE INTO positions(symbol, qty, entry_px, liq_px, ts) VALUES (?,?,?,?,?)",
            (symbol, qty, entry_px, liq_px, ts),
        )

    async def fetch_positions(
        self,
    ) -> Iterable[Tuple[str, float, float | None, float | None, float]]:
        cur = await asyncio.to_thread(
            self.conn.execute, "SELECT symbol, qty, entry_px, liq_px, ts FROM positions"
        )
        rows = await asyncio.to_thread(cur.fetchall)
        return rows

    async def close(self) -> None:
        await self._queue.put((None, ()))
        await self._writer
        await asyncio.to_thread(self.conn.close)
