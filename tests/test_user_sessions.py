from __future__ import annotations

from app.sessions import InMemorySessionStore, UserSessionManager, make_thread_id


def test_same_session_id_is_isolated_by_user():
    manager = UserSessionManager(InMemorySessionStore())
    a = manager.create("alice", session_id="shared", title="Alice session")
    b = manager.create("bob", session_id="shared", title="Bob session")
    assert a.title == "Alice session"
    assert b.title == "Bob session"
    assert manager.get("alice", "shared").title == "Alice session"
    assert manager.get("bob", "shared").title == "Bob session"
    assert make_thread_id("alice", "shared") != make_thread_id("bob", "shared")


def test_session_lifecycle():
    manager = UserSessionManager(InMemorySessionStore())
    created = manager.create("alice", title="Report review")
    assert created.session_id
    assert len(manager.list("alice")) == 1
    renamed = manager.rename("alice", created.session_id, "Updated title")
    assert renamed.title == "Updated title"
    manager.delete("alice", created.session_id)
    assert manager.get("alice", created.session_id) is None
