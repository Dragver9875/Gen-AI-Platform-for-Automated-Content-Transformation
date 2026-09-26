from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Any


@contextmanager
def postgres_checkpointer(database_url: str, *, setup: bool = False) -> Iterator[Any]:
    """Create a durable LangGraph PostgreSQL checkpointer.

    Use setup=True only during initial deployment/migration, not on every request.
    Requires the optional langgraph-checkpoint-postgres package.
    """
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
    except ImportError as exc:
        raise RuntimeError(
            "Install langgraph-checkpoint-postgres to use PostgreSQL LangGraph persistence"
        ) from exc
    with PostgresSaver.from_conn_string(database_url) as saver:
        if setup:
            saver.setup()
        yield saver
