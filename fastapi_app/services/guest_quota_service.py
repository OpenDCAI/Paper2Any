from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

from fastapi_app.config.settings import settings


class GuestQuotaService:
    def __init__(self) -> None:
        self.db_path = Path(settings.GUEST_USAGE_DB_PATH).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS guest_usage (
                    guest_id TEXT NOT NULL,
                    usage_date TEXT NOT NULL,
                    used INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (guest_id, usage_date)
                )
                """
            )

    @staticmethod
    def today() -> str:
        return datetime.utcnow().date().isoformat()

    def get_usage(self, guest_id: str) -> int:
        if not guest_id:
            return 0
        today = self.today()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT used FROM guest_usage WHERE guest_id = ? AND usage_date = ?",
                (guest_id, today),
            ).fetchone()
        return int(row[0]) if row else 0

    def consume(self, guest_id: str, amount: int) -> int:
        today = self.today()
        now = datetime.utcnow().isoformat()
        amount = max(0, int(amount))
        with self._connect() as conn:
            current = conn.execute(
                "SELECT used FROM guest_usage WHERE guest_id = ? AND usage_date = ?",
                (guest_id, today),
            ).fetchone()
            used = int(current[0]) if current else 0
            new_used = used + amount
            conn.execute(
                """
                INSERT INTO guest_usage (guest_id, usage_date, used, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(guest_id, usage_date)
                DO UPDATE SET used = excluded.used, updated_at = excluded.updated_at
                """,
                (guest_id, today, new_used, now),
            )
        return new_used
