from __future__ import annotations

from agents.phase4_graph import Phase4Orchestrator, build_phase4_graph
from app.phase12 import Phase12Pipeline
from app.sessions import UserSessionManager
from generation.service import GenerationService


def create_phase4_orchestrator(
    pipeline: Phase12Pipeline,
    generator: GenerationService,
    *,
    default_top_k: int = 5,
    context_max_chars: int = 60000,
    checkpointer=None,
    session_manager: UserSessionManager | None = None,
) -> Phase4Orchestrator:
    graph = build_phase4_graph(
        pipeline,
        generator,
        default_top_k=default_top_k,
        context_max_chars=context_max_chars,
        checkpointer=checkpointer,
    )
    return Phase4Orchestrator(
        graph,
        default_top_k=default_top_k,
        session_manager=session_manager,
        pipeline=pipeline,
    )
