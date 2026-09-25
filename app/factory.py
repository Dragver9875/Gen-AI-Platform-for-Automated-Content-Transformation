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


def build_phase3(settings: Settings, *, checkpointer=None):
    pipeline = build_phase12(settings)
    return create_phase3_orchestrator(
        pipeline,
        default_top_k=settings.phase3_default_top_k,
        context_max_chars=settings.phase3_context_max_chars,
        checkpointer=checkpointer,
    )
