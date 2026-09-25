from __future__ import annotations

from app.config import Settings
from database.chroma import ChromaCloudStore
from ingestion.chunker import StructureAwareChunker
from ingestion.preflight import PdfPreflight
from ingestion.router import IngestionRouter
from providers.docling import DoclingAPIProvider
from providers.harrier import HarrierEmbeddingProvider
from providers.reranker import HostedReranker
from providers.siglip import SigLIPRoutingProvider
from providers.vlm import VLMProvider
from retrieval.hybrid_retrieval import HybridRetriever
from app.phase12 import Phase12Pipeline
from app.phase3 import create_phase3_orchestrator
from app.phase4 import create_phase4_orchestrator
from app.sessions import InMemorySessionStore, PostgresSessionStore, UserSessionManager
from generation.service import GenerationService
from providers.llm import HostedLLMProvider


def build_phase12(settings: Settings):
    docling = DoclingAPIProvider(
        settings.docling_api_url,
        settings.docling_api_key,
        timeout_s=settings.docling_timeout_s,
        retries=settings.http_retries,
    )
    siglip = SigLIPRoutingProvider(
        settings.siglip_api_url,
        settings.siglip_api_key,
        timeout_s=settings.http_timeout_s,
        retries=settings.http_retries,
    )
    vlm = VLMProvider(
        settings.vlm_api_url,
        settings.vlm_api_key,
        timeout_s=settings.http_timeout_s,
        retries=settings.http_retries,
    )
    router = IngestionRouter(
        docling=docling,
        siglip=siglip,
        vlm=vlm,
        pdf_preflight=PdfPreflight(
            native_text_chars=settings.pdf_native_text_chars,
            image_coverage_threshold=settings.pdf_image_coverage_threshold,
        ),
        enable_pdf_visual_fallback=settings.pdf_visual_fallback_enabled,
        pdf_visual_fallback_max_pages=settings.pdf_visual_fallback_max_pages,
        document_like_image_labels=set(settings.siglip_document_labels),
        pdf_visual_prompt=settings.pdf_visual_prompt,
    )
    chunker = StructureAwareChunker(
        target_chars=settings.chunk_target_chars,
        overlap_chars=settings.chunk_overlap_chars,
    )
    embedder = HarrierEmbeddingProvider(
        settings.harrier_api_url,
        settings.harrier_api_key,
        api_style=settings.harrier_api_style,
        batch_size=settings.harrier_batch_size,
        query_instruction=settings.harrier_query_instruction,
        timeout_s=settings.http_timeout_s,
        retries=settings.http_retries,
    )
    store = ChromaCloudStore(
        settings.chroma_api_key,
        settings.chroma_tenant,
        settings.chroma_database,
        settings.chroma_collection,
    )
    reranker = (
        HostedReranker(settings.reranker_api_url, settings.reranker_api_key, timeout_s=settings.http_timeout_s, retries=settings.http_retries)
        if settings.reranker_api_url
        else None
    )
    retriever = HybridRetriever(store, embedder, reranker)
    return Phase12Pipeline(router, chunker, retriever)


def build_session_manager(settings: Settings) -> UserSessionManager:
    backend = settings.session_store_backend.lower()
    if backend == "memory":
        return UserSessionManager(InMemorySessionStore())
    if backend == "postgres":
        if not settings.session_database_url:
            raise ValueError("SESSION_DATABASE_URL is required when SESSION_STORE_BACKEND=postgres")
        return UserSessionManager(PostgresSessionStore(settings.session_database_url, table=settings.session_table))
    raise ValueError(f"Unsupported SESSION_STORE_BACKEND: {settings.session_store_backend}")


def build_phase3(settings: Settings, *, checkpointer=None):
    pipeline = build_phase12(settings)
    sessions = build_session_manager(settings)
    return create_phase3_orchestrator(
        pipeline,
        default_top_k=settings.phase3_default_top_k,
        context_max_chars=settings.phase3_context_max_chars,
        checkpointer=checkpointer,
        session_manager=sessions,
    )


def build_phase4(settings: Settings, *, checkpointer=None):
    if not settings.llm_api_url or not settings.llm_api_key:
        raise ValueError("LLM_API_URL and LLM_API_KEY are required for Phase 4")
    pipeline = build_phase12(settings)
    sessions = build_session_manager(settings)
    llm = HostedLLMProvider(
        settings.llm_api_url,
        settings.llm_api_key,
        api_style=settings.llm_api_style,
        model=settings.llm_model,
        response_mode=settings.llm_response_mode,
        timeout_s=settings.http_timeout_s,
        retries=settings.http_retries,
    )
    generator = GenerationService(
        llm,
        temperature=settings.phase4_temperature,
        max_tokens=settings.phase4_max_tokens,
        group_context_max_chars=settings.phase4_group_context_max_chars,
        digest_max_tokens=settings.phase4_digest_max_tokens,
    )
    return create_phase4_orchestrator(
        pipeline,
        generator,
        default_top_k=settings.phase3_default_top_k,
        context_max_chars=settings.phase3_context_max_chars,
        checkpointer=checkpointer,
        session_manager=sessions,
    )
