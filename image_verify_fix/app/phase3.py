from __future__ import annotations

from agents.graph import Phase3Orchestrator, build_phase3_graph
from app.phase12 import Phase12Pipeline
from app.sessions import UserSessionManager


def create_phase3_orchestrator(
    pipeline: Phase12Pipeline, *, default_top_k: int = 5, context_max_chars: int = 60000, checkpointer=None, session_manager: UserSessionManager | None = None
) -> Phase3Orchestrator:
    graph = build_phase3_graph(pipeline, default_top_k=default_top_k, context_max_chars=context_max_chars, checkpointer=checkpointer)
    return Phase3Orchestrator(graph, default_top_k=default_top_k, session_manager=session_manager, pipeline=pipeline)
