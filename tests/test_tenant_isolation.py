from __future__ import annotations

from agents.nodes import _where


def test_where_always_scopes_user_and_session():
    assert _where("u1", "s1") == {"$and": [{"user_id": "u1"}, {"session_id": "s1"}]}
    assert _where("u1", "s1", ["src"]) == {"$and": [
        {"user_id": "u1"}, {"session_id": "s1"}, {"source_id": "src"}
    ]}
