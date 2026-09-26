from __future__ import annotations

from agents.phase5_graph import Phase5Orchestrator, build_phase5_graph
from app.phase12 import Phase12Pipeline
from app.sessions import UserSessionManager
from generation.service import GenerationService
from verification.service import VerificationService


def create_phase5_orchestrator(
    pipeline: Phase12Pipeline,
    generator: GenerationService,
    verifier: VerificationService,
    *,
    default_top_k: int = 5,
    context_max_chars: int = 60000,
    checkpointer=None,
    session_manager: UserSessionManager | None = None,
) -> Phase5Orchestrator:
    graph = build_phase5_graph(
        pipeline,
        generator,
        verifier,
        default_top_k=default_top_k,
        context_max_chars=context_max_chars,
        checkpointer=checkpointer,
    )
    return Phase5Orchestrator(
        graph,
        default_top_k=default_top_k,
        session_manager=session_manager,
        pipeline=pipeline,
        max_repair_attempts=verifier.max_repair_attempts,
    )
