# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_PLACEMENTS = frozenset({"local", "cloud", "sleeping"})


class SessionConflictError(RuntimeError):
    """Raised when a compare-and-swap update or lease cannot be acquired."""


@dataclass(frozen=True)
class SessionRecord:
    session_id: str
    placement: str = "sleeping"
    endpoint: str = ""
    model_id: str = ""
    model_revision: str = ""
    state_uri: str = ""
    state_sha256: str = ""
    generation: int = 0
    updated_at: float = 0.0
    lease_owner: str = ""
    lease_expires_at: float = 0.0

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("session_id must not be empty")
        if self.placement not in _PLACEMENTS:
            raise ValueError(
                f"placement must be one of {sorted(_PLACEMENTS)}, "
                f"got {self.placement!r}"
            )


class SessionRegistry:
    """Small SQLite registry with optimistic concurrency and expiring leases."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).expanduser().resolve().parent.mkdir(
                parents=True, exist_ok=True
            )
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self.path,
            check_same_thread=False,
            isolation_level=None,
        )
        self._connection.row_factory = sqlite3.Row
        with self._lock:
            if self.path != ":memory:":
                self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA busy_timeout=5000")
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    placement TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    model_revision TEXT NOT NULL,
                    state_uri TEXT NOT NULL,
                    state_sha256 TEXT NOT NULL,
                    generation INTEGER NOT NULL,
                    updated_at REAL NOT NULL,
                    lease_owner TEXT NOT NULL,
                    lease_expires_at REAL NOT NULL
                )
                """
            )

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> SessionRegistry:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    @staticmethod
    def _from_row(row: sqlite3.Row | None) -> SessionRecord | None:
        return None if row is None else SessionRecord(**dict(row))

    def get(self, session_id: str) -> SessionRecord | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        return self._from_row(row)

    def list(self, *, limit: int = 1000) -> list[SessionRecord]:
        if limit < 1:
            return []
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [SessionRecord(**dict(row)) for row in rows]

    def upsert(
        self,
        record: SessionRecord,
        *,
        expected_generation: int | None = None,
        now: float | None = None,
    ) -> SessionRecord:
        """Insert or update a session and increment its generation.

        expected_generation=0 means the caller expects the row not to exist.
        """
        values: dict[str, Any] = asdict(record)
        timestamp = time.time() if now is None else float(now)
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                current = self._connection.execute(
                    "SELECT generation FROM sessions WHERE session_id = ?",
                    (record.session_id,),
                ).fetchone()
                actual_generation = 0 if current is None else int(current[0])
                if (
                    expected_generation is not None
                    and actual_generation != expected_generation
                ):
                    raise SessionConflictError(
                        f"session {record.session_id!r} generation changed: "
                        f"expected {expected_generation}, got {actual_generation}"
                    )
                values["generation"] = actual_generation + 1
                values["updated_at"] = timestamp
                columns = tuple(values)
                placeholders = ", ".join("?" for _ in columns)
                updates = ", ".join(
                    f"{column}=excluded.{column}"
                    for column in columns
                    if column != "session_id"
                )
                self._connection.execute(
                    f"""
                    INSERT INTO sessions ({", ".join(columns)})
                    VALUES ({placeholders})
                    ON CONFLICT(session_id) DO UPDATE SET {updates}
                    """,
                    tuple(values[column] for column in columns),
                )
                self._connection.execute("COMMIT")
            except Exception:
                self._connection.execute("ROLLBACK")
                raise
        result = self.get(record.session_id)
        assert result is not None
        return result

    def claim(
        self,
        session_id: str,
        owner: str,
        *,
        ttl: float = 30.0,
        now: float | None = None,
    ) -> SessionRecord:
        if not owner:
            raise ValueError("lease owner must not be empty")
        if ttl <= 0:
            raise ValueError("lease ttl must be positive")
        timestamp = time.time() if now is None else float(now)
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._connection.execute(
                    "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
                ).fetchone()
                if row is None:
                    raise KeyError(session_id)
                current = SessionRecord(**dict(row))
                if (
                    current.lease_owner
                    and current.lease_owner != owner
                    and current.lease_expires_at > timestamp
                ):
                    raise SessionConflictError(
                        f"session {session_id!r} is leased by "
                        f"{current.lease_owner!r} until {current.lease_expires_at}"
                    )
                self._connection.execute(
                    """
                    UPDATE sessions
                    SET lease_owner = ?, lease_expires_at = ?,
                        generation = generation + 1, updated_at = ?
                    WHERE session_id = ?
                    """,
                    (owner, timestamp + ttl, timestamp, session_id),
                )
                self._connection.execute("COMMIT")
            except Exception:
                self._connection.execute("ROLLBACK")
                raise
        claimed = self.get(session_id)
        assert claimed is not None
        return claimed

    def release(
        self,
        session_id: str,
        owner: str,
        *,
        now: float | None = None,
    ) -> SessionRecord:
        timestamp = time.time() if now is None else float(now)
        with self._lock:
            cursor = self._connection.execute(
                """
                UPDATE sessions
                SET lease_owner = '', lease_expires_at = 0,
                    generation = generation + 1, updated_at = ?
                WHERE session_id = ? AND lease_owner = ?
                """,
                (timestamp, session_id, owner),
            )
        if cursor.rowcount != 1:
            raise SessionConflictError(
                f"session {session_id!r} is not leased by {owner!r}"
            )
        released = self.get(session_id)
        assert released is not None
        return released


__all__ = ["SessionConflictError", "SessionRecord", "SessionRegistry"]
