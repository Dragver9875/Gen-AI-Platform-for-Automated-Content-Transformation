from __future__ import annotations

import hashlib
import json
import re
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_thread_id(user_id: str, session_id: str) -> str:
    """Return a stable, non-PII LangGraph thread id for a user/session pair."""
    raw = f"{user_id}\x1f{session_id}".encode("utf-8")
    return "usr_" + hashlib.sha256(raw).hexdigest()[:40]


@dataclass
class UserSession:
    user_id: str
    session_id: str
    title: str = "Untitled session"
    created_at: str = field(default_factory=utcnow_iso)
    updated_at: str = field(default_factory=utcnow_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": dict(self.metadata),
        }


class SessionStore(Protocol):
    def upsert(self, session: UserSession) -> UserSession: ...
    def get(self, user_id: str, session_id: str) -> UserSession | None: ...
    def list(self, user_id: str, *, limit: int = 100, offset: int = 0) -> list[UserSession]: ...
    def delete(self, user_id: str, session_id: str) -> None: ...


class InMemorySessionStore:
    """Development/test session store. Do not use for multi-instance production."""

    def __init__(self):
        self._lock = threading.RLock()
        self._data: dict[tuple[str, str], UserSession] = {}

    def upsert(self, session: UserSession) -> UserSession:
        with self._lock:
            self._data[(session.user_id, session.session_id)] = session
        return session

    def get(self, user_id: str, session_id: str) -> UserSession | None:
        with self._lock:
            return self._data.get((user_id, session_id))

    def list(self, user_id: str, *, limit: int = 100, offset: int = 0) -> list[UserSession]:
        with self._lock:
            rows = [s for (uid, _), s in self._data.items() if uid == user_id]
        rows.sort(key=lambda s: s.updated_at, reverse=True)
        return rows[offset : offset + limit]

    def delete(self, user_id: str, session_id: str) -> None:
        with self._lock:
            self._data.pop((user_id, session_id), None)


class PostgresSessionStore:
    """Durable production session store backed by PostgreSQL.

    The psycopg dependency is imported lazily so development/test environments can
    use the in-memory store without requiring a local PostgreSQL client.
    """

    def __init__(self, database_url: str, *, table: str = "user_sessions"):
        if not database_url:
            raise ValueError("database_url is required for PostgresSessionStore")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
            raise ValueError("SESSION_TABLE must be a simple SQL identifier")
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError("psycopg is required for PostgreSQL session persistence") from exc
        self.psycopg = psycopg
        self.database_url = database_url
        self.table = table
        self._ensure_schema()

    def _connect(self):
        return self.psycopg.connect(self.database_url)

    def _ensure_schema(self) -> None:
        sql = f"""
        CREATE TABLE IF NOT EXISTS {self.table} (
            user_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            title TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL,
            metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            PRIMARY KEY (user_id, session_id)
        );
        CREATE INDEX IF NOT EXISTS {self.table}_user_updated_idx
            ON {self.table} (user_id, updated_at DESC);
        """
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql)

    def upsert(self, session: UserSession) -> UserSession:
        sql = f"""
        INSERT INTO {self.table} (user_id, session_id, title, created_at, updated_at, metadata)
        VALUES (%s, %s, %s, %s, %s, %s::jsonb)
        ON CONFLICT (user_id, session_id) DO UPDATE SET
            title = EXCLUDED.title,
            updated_at = EXCLUDED.updated_at,
            metadata = EXCLUDED.metadata
        """
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, (
                session.user_id, session.session_id, session.title,
                session.created_at, session.updated_at, json.dumps(session.metadata),
            ))
        return session

    def get(self, user_id: str, session_id: str) -> UserSession | None:
        sql = f"SELECT user_id, session_id, title, created_at, updated_at, metadata FROM {self.table} WHERE user_id=%s AND session_id=%s"
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, (user_id, session_id))
            row = cur.fetchone()
        if not row:
            return None
        return UserSession(
            user_id=row[0], session_id=row[1], title=row[2],
            created_at=row[3].isoformat(), updated_at=row[4].isoformat(), metadata=row[5] or {},
        )

    def list(self, user_id: str, *, limit: int = 100, offset: int = 0) -> list[UserSession]:
        sql = f"""SELECT user_id, session_id, title, created_at, updated_at, metadata
                  FROM {self.table} WHERE user_id=%s ORDER BY updated_at DESC LIMIT %s OFFSET %s"""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, (user_id, limit, offset))
            rows = cur.fetchall()
        return [UserSession(
            user_id=r[0], session_id=r[1], title=r[2],
            created_at=r[3].isoformat(), updated_at=r[4].isoformat(), metadata=r[5] or {},
        ) for r in rows]

    def delete(self, user_id: str, session_id: str) -> None:
        sql = f"DELETE FROM {self.table} WHERE user_id=%s AND session_id=%s"
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, (user_id, session_id))


class UserSessionManager:
    def __init__(self, store: SessionStore):
        self.store = store

    def create(
        self,
        user_id: str,
        *,
        session_id: str | None = None,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> UserSession:
        user_id = user_id.strip()
        if not user_id:
            raise ValueError("user_id is required")
        session_id = (session_id or uuid.uuid4().hex).strip()
        if not session_id:
            raise ValueError("session_id is required")
        existing = self.store.get(user_id, session_id)
        if existing:
            return existing
        now = utcnow_iso()
        return self.store.upsert(UserSession(
            user_id=user_id,
            session_id=session_id,
            title=(title or "Untitled session").strip() or "Untitled session",
            created_at=now,
            updated_at=now,
            metadata=dict(metadata or {}),
        ))

    def ensure(self, user_id: str, session_id: str, *, title: str | None = None) -> UserSession:
        existing = self.store.get(user_id, session_id)
        if existing:
            existing.updated_at = utcnow_iso()
            if title:
                existing.title = title.strip() or existing.title
            return self.store.upsert(existing)
        return self.create(user_id, session_id=session_id, title=title)

    def get(self, user_id: str, session_id: str) -> UserSession | None:
        return self.store.get(user_id, session_id)

    def list(self, user_id: str, *, limit: int = 100, offset: int = 0) -> list[UserSession]:
        return self.store.list(user_id, limit=limit, offset=offset)

    def rename(self, user_id: str, session_id: str, title: str) -> UserSession:
        session = self.store.get(user_id, session_id)
        if not session:
            raise KeyError(f"Unknown session: {session_id}")
        session.title = title.strip() or session.title
        session.updated_at = utcnow_iso()
        return self.store.upsert(session)

    def delete(self, user_id: str, session_id: str) -> None:
        self.store.delete(user_id, session_id)
