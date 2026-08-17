from __future__ import annotations

from pathlib import Path
import sqlite3
import time


class MemoryStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at REAL NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user','assistant')),
                    content TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def add_turn(self, user_text: str, assistant_text: str) -> None:
        now = time.time()
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO messages(created_at, role, content) VALUES (?, ?, ?)",
                [
                    (now, "user", user_text),
                    (now + 0.000001, "assistant", assistant_text),
                ],
            )
            conn.commit()

    def recent_messages(self, turns: int) -> list[dict[str, str]]:
        limit = max(0, int(turns)) * 2
        if limit == 0:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content FROM messages ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        rows.reverse()
        return [{"role": role, "content": content} for role, content in rows]

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM messages")
            conn.commit()
