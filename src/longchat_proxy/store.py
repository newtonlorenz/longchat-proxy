from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .messages import canonical_hash


@dataclass(frozen=True, slots=True)
class SummaryRecord:
    prefix_hash: str
    prefix_count: int
    summary: str
    conversation_key: str = ""


class SummaryStore(Protocol):
    def find_best(self, turns: list[dict], conversation_key: str = "") -> SummaryRecord | None: ...

    def put(self, record: SummaryRecord) -> None: ...

    def count(self) -> int: ...

    def close(self) -> None: ...


class NullStore:
    def find_best(self, turns: list[dict], conversation_key: str = "") -> None:
        return None

    def put(self, record: SummaryRecord) -> None:
        return None

    def count(self) -> int:
        return 0

    def close(self) -> None:
        return None


class MemoryStore:
    def __init__(self, max_records: int = 500) -> None:
        self.max_records = max_records
        self._records: list[SummaryRecord] = []
        self._lock = threading.Lock()

    def find_best(self, turns: list[dict], conversation_key: str = "") -> SummaryRecord | None:
        with self._lock:
            candidates = sorted(self._records, key=lambda item: item.prefix_count, reverse=True)
        for record in candidates:
            if record.prefix_count > len(turns):
                continue
            if conversation_key and record.conversation_key not in {"", conversation_key}:
                continue
            if canonical_hash(turns, record.prefix_count) == record.prefix_hash:
                return record
        return None

    def put(self, record: SummaryRecord) -> None:
        with self._lock:
            self._records = [
                item
                for item in self._records
                if not (
                    item.prefix_hash == record.prefix_hash
                    and item.conversation_key == record.conversation_key
                )
            ]
            self._records.append(record)
            self._records = self._records[-self.max_records :]

    def count(self) -> int:
        with self._lock:
            return len(self._records)

    def close(self) -> None:
        return None


class SQLiteStore:
    def __init__(self, path: Path, max_records: int = 500, ttl_days: int = 30) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.max_records = max_records
        self.ttl_days = ttl_days
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS summaries (
                prefix_hash TEXT NOT NULL,
                prefix_count INTEGER NOT NULL,
                summary TEXT NOT NULL,
                conversation_key TEXT NOT NULL DEFAULT '',
                created_at INTEGER NOT NULL,
                accessed_at INTEGER NOT NULL,
                PRIMARY KEY (prefix_hash, conversation_key)
            )
            """
        )
        self._connection.commit()
        self._prune()

    def _prune(self) -> None:
        now = int(time.time())
        cutoff = now - self.ttl_days * 86400
        with self._lock:
            self._connection.execute("DELETE FROM summaries WHERE accessed_at < ?", (cutoff,))
            self._connection.execute(
                """
                DELETE FROM summaries WHERE rowid IN (
                    SELECT rowid FROM summaries ORDER BY accessed_at DESC LIMIT -1 OFFSET ?
                )
                """,
                (self.max_records,),
            )
            self._connection.commit()

    def find_best(self, turns: list[dict], conversation_key: str = "") -> SummaryRecord | None:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT prefix_hash, prefix_count, summary, conversation_key
                FROM summaries
                WHERE prefix_count <= ? AND (conversation_key = '' OR conversation_key = ?)
                ORDER BY prefix_count DESC LIMIT 100
                """,
                (len(turns), conversation_key),
            ).fetchall()
        for row in rows:
            record = SummaryRecord(*row)
            if canonical_hash(turns, record.prefix_count) == record.prefix_hash:
                with self._lock:
                    self._connection.execute(
                        "UPDATE summaries SET accessed_at = ? WHERE prefix_hash = ? "
                        "AND conversation_key = ?",
                        (int(time.time()), record.prefix_hash, record.conversation_key),
                    )
                    self._connection.commit()
                return record
        return None

    def put(self, record: SummaryRecord) -> None:
        now = int(time.time())
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO summaries
                    (prefix_hash, prefix_count, summary, conversation_key, created_at, accessed_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(prefix_hash, conversation_key) DO UPDATE SET
                    prefix_count = excluded.prefix_count,
                    summary = excluded.summary,
                    accessed_at = excluded.accessed_at
                """,
                (
                    record.prefix_hash,
                    record.prefix_count,
                    record.summary,
                    record.conversation_key,
                    now,
                    now,
                ),
            )
            self._connection.commit()
        self._prune()

    def count(self) -> int:
        with self._lock:
            return int(self._connection.execute("SELECT COUNT(*) FROM summaries").fetchone()[0])

    def close(self) -> None:
        with self._lock:
            self._connection.close()


def create_store(mode: str, path: Path, max_records: int, ttl_days: int) -> SummaryStore:
    if mode == "off":
        return NullStore()
    if mode == "sqlite":
        return SQLiteStore(path, max_records=max_records, ttl_days=ttl_days)
    return MemoryStore(max_records=max_records)
