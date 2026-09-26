from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from agents.nodes import Phase3Nodes
from agents.state import AgentState
from app.phase12 import Phase12Pipeline
from app.sessions import UserSessionManager, make_thread_id


def _route_after_initialize(state: AgentState) -> str:
    if state.get("status") == "error":
        return "error"
    return "ingest" if state.get("pending_source_paths") else "classify"


def _route_intent(state: AgentState) -> str:
    intent = state.get("detected_intent")
    if intent == "qa": return "qa"
    if intent == "transform": return "transform"
    if intent == "index_only": return "index_only"
    return "error"


def build_phase3_graph(
    pipeline: Phase12Pipeline, *, default_top_k: int = 5, context_max_chars: int = 60000, checkpointer: Any | None = None
):
    nodes = Phase3Nodes(pipeline, default_top_k=default_top_k, context_max_chars=context_max_chars)
    builder = StateGraph(AgentState)
    builder.add_node("initialize", nodes.initialize)
    builder.add_node("ingest_sources", nodes.ingest_sources)
    builder.add_node("classify_intent", nodes.classify_intent)
    builder.add_node("retrieve_qa", nodes.retrieve_qa)
    builder.add_node("retrieve_transform", nodes.retrieve_transform)
    builder.add_node("build_context", nodes.build_context)
    builder.add_node("finalize_index_only", nodes.finalize_index_only)
    builder.add_node("finalize_error", nodes.finalize_error)
    builder.add_edge(START, "initialize")
    builder.add_conditional_edges("initialize", _route_after_initialize, {"ingest": "ingest_sources", "classify": "classify_intent", "error": "finalize_error"})
    builder.add_edge("ingest_sources", "classify_intent")
    builder.add_conditional_edges("classify_intent", _route_intent, {"qa": "retrieve_qa", "transform": "retrieve_transform", "index_only": "finalize_index_only", "error": "finalize_error"})
    builder.add_edge("retrieve_qa", "build_context")
    builder.add_edge("retrieve_transform", "build_context")
    builder.add_edge("build_context", END)
    builder.add_edge("finalize_index_only", END)
    builder.add_edge("finalize_error", END)
    return builder.compile(checkpointer=checkpointer or InMemorySaver(), name="phase3-orchestrator")


class Phase3Orchestrator:
    """User/session-aware façade around the compiled LangGraph."""

    def __init__(self, graph, *, default_top_k: int = 5, session_manager: UserSessionManager | None = None, pipeline: Phase12Pipeline | None = None):
        self.graph = graph
        self.default_top_k = default_top_k
        self.session_manager = session_manager
        self.pipeline = pipeline

    def create_session(self, user_id: str, *, session_id: str | None = None, title: str | None = None, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.session_manager:
            raise RuntimeError("No session manager configured")
        return self.session_manager.create(user_id, session_id=session_id, title=title, metadata=metadata).to_dict()

    def list_sessions(self, user_id: str, *, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        if not self.session_manager:
            raise RuntimeError("No session manager configured")
        return [s.to_dict() for s in self.session_manager.list(user_id, limit=limit, offset=offset)]

    def rename_session(self, user_id: str, session_id: str, title: str) -> dict[str, Any]:
        if not self.session_manager:
            raise RuntimeError("No session manager configured")
        return self.session_manager.rename(user_id, session_id, title).to_dict()

    def delete_session(self, user_id: str, session_id: str, *, delete_vectors: bool = True) -> None:
        if delete_vectors and self.pipeline is not None:
            store = getattr(self.pipeline.retriever, "store", None)
            if store is not None and hasattr(store, "delete_session"):
                store.delete_session(user_id, session_id)
        if self.session_manager:
            self.session_manager.delete(user_id, session_id)

    def invoke(
        self, *, user_id: str, session_id: str | None = None, source_paths: list[str] | None = None, query: str = "",
        request_mode: str = "auto", selected_source_ids: list[str] | None = None, top_k: int | None = None, session_title: str | None = None
    ) -> AgentState:
        user_id = user_id.strip()
        if not user_id:
            raise ValueError("user_id is required")
        if session_id is None:
            if not self.session_manager:
                raise ValueError("session_id is required when no session manager is configured")
            session_id = self.session_manager.create(user_id, title=session_title).session_id
        elif self.session_manager:
            self.session_manager.ensure(user_id, session_id, title=session_title)

        payload: AgentState = {
            "user_id": user_id,
            "session_id": session_id,
            "query": query,
            "request_mode": request_mode,
            "pending_source_paths": list(source_paths or []),
            "selected_source_ids": list(selected_source_ids or []),
            "top_k": int(top_k or self.default_top_k),
            "retrieved_documents": [],
            "context_groups": [],
            "prepared_context": "",
            "warnings": [],
            "errors": [],
        }
        config = {"configurable": {"thread_id": make_thread_id(user_id, session_id)}}
        return self.graph.invoke(payload, config=config)

    def get_state(self, user_id: str, session_id: str):
        return self.graph.get_state({"configurable": {"thread_id": make_thread_id(user_id, session_id)}})
