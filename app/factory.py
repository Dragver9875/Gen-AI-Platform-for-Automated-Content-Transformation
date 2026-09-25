from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.phase12 import Phase12Pipeline
from app.phase3 import create_phase3_orchestrator
from app.phase4 import create_phase4_orchestrator
from app.phase5 import create_phase5_orchestrator
from app.phase6 import create_phase6_orchestrator
from app.sessions import InMemorySessionStore, PostgresSessionStore, UserSessionManager
from artifacts.generators import (
    CreativeImageArtifactGenerator,
    PDFArtifactGenerator,
    PPTXArtifactGenerator,
    SVGArtifactGenerator,
    TextArtifactGenerator,
)
from artifacts.registry import ArtifactRegistry
from artifacts.service import ArtifactService
from database.chroma import ChromaCloudStore
from generation.service import GenerationService
from ingestion.chunker import StructureAwareChunker
from ingestion.preflight import PdfPreflight
from ingestion.router import IngestionRouter
from providers.docling import DoclingAPIProvider
from providers.harrier import HarrierEmbeddingProvider
from providers.image_generation import HostedImageGenerationProvider
from providers.llm import HostedLLMProvider
from providers.registry import ProviderCapability, ProviderRegistry
from providers.reranker import HostedReranker
from providers.siglip import SigLIPRoutingProvider
from providers.vlm import VLMProvider
from retrieval.config import RetrievalConfig
from retrieval.hybrid_retrieval import HybridRetriever
from verification.profiles import VerificationProfileRegistry
from verification.service import VerificationService


def build_provider_registry(settings: Settings, *, include_generation: bool = False, include_image_generation: bool = False) -> ProviderRegistry:
    registry = ProviderRegistry()
    registry.register(
        ProviderCapability.DOCUMENT_UNDERSTANDING,
        DoclingAPIProvider(
            settings.docling_api_url,
            settings.docling_api_key,
            model=settings.docling_model,
            timeout_s=settings.docling_timeout_s,
            retries=settings.http_retries,
            render_dpi=settings.docling_render_dpi,
        ),
        name="docling",
    )
    registry.register(
        ProviderCapability.VISUAL_ROUTING,
        SigLIPRoutingProvider(
            settings.siglip_api_url,
            settings.siglip_api_key,
            timeout_s=settings.http_timeout_s,
            retries=settings.http_retries,
        ),
        name="siglip",
    )
    registry.register(
        ProviderCapability.VISUAL_UNDERSTANDING,
        VLMProvider(
            settings.vlm_api_url,
            settings.vlm_api_key,
            timeout_s=settings.http_timeout_s,
            retries=settings.http_retries,
        ),
        name="vlm",
    )
    registry.register(
        ProviderCapability.EMBEDDING,
        HarrierEmbeddingProvider(
            settings.harrier_api_url,
            settings.harrier_api_key,
            api_style=settings.harrier_api_style,
            model=settings.harrier_model,
            batch_size=settings.harrier_batch_size,
            prompt_name=settings.harrier_prompt_name,
            normalize=settings.harrier_normalize,
            query_instruction=settings.harrier_query_instruction,
            timeout_s=settings.http_timeout_s,
            retries=settings.http_retries,
        ),
        name="harrier",
    )
    if settings.reranker_api_url:
        registry.register(
            ProviderCapability.RERANKING,
            HostedReranker(
                settings.reranker_api_url,
                settings.reranker_api_key or settings.hf_token or "",
                api_style=settings.reranker_api_style,
                timeout_s=settings.http_timeout_s,
                retries=settings.http_retries,
            ),
            name="hosted",
        )
    if include_generation:
        if not settings.llm_api_url or not settings.llm_api_key:
            raise ValueError("LLM_API_URL and LLM_API_KEY are required for generation phases")
        llm = HostedLLMProvider(
            settings.llm_api_url,
            settings.llm_api_key,
            api_style=settings.llm_api_style,
            model=settings.llm_model,
            response_mode=settings.llm_response_mode,
            timeout_s=settings.http_timeout_s,
            retries=settings.http_retries,
        )
        registry.register(ProviderCapability.STRUCTURED_GENERATION, llm, name="primary")
        registry.register(ProviderCapability.VERIFICATION, llm, name="primary")
        if settings.verifier_api_url:
            registry.register(
                ProviderCapability.VERIFICATION,
                HostedLLMProvider(
                    settings.verifier_api_url,
                    settings.verifier_api_key or settings.hf_token or settings.llm_api_key or "",
                    api_style=settings.verifier_api_style,
                    model=settings.verifier_model,
                    response_mode=settings.verifier_response_mode,
                    timeout_s=settings.http_timeout_s,
                    retries=settings.http_retries,
                ),
                name="verifier",
                default=True,
            )
    if include_image_generation and settings.image_gen_api_url and settings.image_gen_api_key:
        registry.register(
            ProviderCapability.IMAGE_GENERATION,
            HostedImageGenerationProvider(
                settings.image_gen_api_url,
                settings.image_gen_api_key,
                model=settings.image_gen_model,
                api_style=settings.image_gen_api_style,
                timeout_s=max(settings.http_timeout_s, 180.0),
                retries=settings.http_retries,
            ),
            name="hosted",
        )
    return registry


def build_phase12(settings: Settings, *, providers: ProviderRegistry | None = None):
    providers = providers or build_provider_registry(settings)
    docling = providers.resolve(ProviderCapability.DOCUMENT_UNDERSTANDING)
    siglip = providers.resolve(ProviderCapability.VISUAL_ROUTING)
    vlm = providers.resolve(ProviderCapability.VISUAL_UNDERSTANDING)
    embedder = providers.resolve(ProviderCapability.EMBEDDING)
    reranker = providers.resolve(ProviderCapability.RERANKING, required=False)

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
    store = ChromaCloudStore(
        settings.chroma_api_key,
        settings.chroma_tenant,
        settings.chroma_database,
        settings.chroma_collection,
    )
    retrieval_config = RetrievalConfig(
        strategy=settings.retrieval_strategy,
        bm25_k=settings.retrieval_bm25_k,
        vector_k=settings.retrieval_vector_k,
        final_k=settings.phase3_default_top_k,
        rrf_k=settings.retrieval_rrf_k,
        use_reranker=settings.retrieval_use_reranker,
    )
    retriever = HybridRetriever(store, embedder, reranker, default_config=retrieval_config)
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


def _build_generation_service(settings: Settings, providers: ProviderRegistry) -> GenerationService:
    llm = providers.resolve(ProviderCapability.STRUCTURED_GENERATION)
    return GenerationService(
        llm,
        temperature=settings.phase4_temperature,
        max_tokens=settings.phase4_max_tokens,
        group_context_max_chars=settings.phase4_group_context_max_chars,
        digest_max_tokens=settings.phase4_digest_max_tokens,
    )


def _build_verification_service(settings: Settings, providers: ProviderRegistry) -> VerificationService:
    generator = providers.resolve(ProviderCapability.STRUCTURED_GENERATION)
    verifier_provider = providers.resolve(ProviderCapability.VERIFICATION)
    profile_registry = VerificationProfileRegistry.from_json(
        settings.phase5_verification_profiles_json,
        default_profile=settings.phase5_default_verification_profile,
    )
    return VerificationService(
        verifier_provider,
        repair_generator=generator,
        verification_temperature=settings.phase5_verification_temperature,
        verification_max_tokens=settings.phase5_verification_max_tokens,
        repair_temperature=settings.phase5_repair_temperature,
        repair_max_tokens=settings.phase5_repair_max_tokens,
        max_repair_attempts=settings.phase5_max_repair_attempts,
        min_faithfulness_score=settings.phase5_min_faithfulness_score,
        max_claims_per_call=settings.phase5_max_claims_per_call,
        evidence_chars_per_claim=settings.phase5_evidence_chars_per_claim,
        profile_registry=profile_registry,
    )


def build_artifact_service(settings: Settings, providers: ProviderRegistry) -> ArtifactService:
    llm = providers.resolve(ProviderCapability.STRUCTURED_GENERATION)
    image_provider = providers.resolve(ProviderCapability.IMAGE_GENERATION, required=False)
    registry = ArtifactRegistry()
    registry.register_generator("text", TextArtifactGenerator(llm, temperature=settings.phase6_temperature, max_tokens=settings.phase6_max_tokens))
    registry.register_generator(
        "pdf",
        PDFArtifactGenerator(
            llm,
            temperature=settings.phase6_temperature,
            max_tokens=settings.phase6_max_tokens,
            typst_binary=settings.phase6_typst_binary,
            retain_source=settings.phase6_retain_typst_source,
        ),
    )
    registry.register_generator("pptx", PPTXArtifactGenerator(llm, temperature=settings.phase6_temperature, max_tokens=settings.phase6_max_tokens))
    registry.register_generator("svg", SVGArtifactGenerator(llm, temperature=settings.phase6_temperature, max_tokens=settings.phase6_max_tokens))
    registry.register_generator("creative_image", CreativeImageArtifactGenerator(llm, image_provider, temperature=settings.phase6_temperature, max_tokens=settings.phase6_max_tokens))
    spec_path = Path(settings.phase6_artifact_specs_path)
    if not spec_path.is_absolute():
        spec_path = Path(__file__).resolve().parents[1] / spec_path
    registry.load_specs(spec_path)
    return ArtifactService(registry, output_dir=settings.phase6_output_dir, fail_fast=settings.phase6_fail_fast)


def build_phase3(settings: Settings, *, checkpointer=None):
    providers = build_provider_registry(settings)
    pipeline = build_phase12(settings, providers=providers)
    return create_phase3_orchestrator(
        pipeline,
        default_top_k=settings.phase3_default_top_k,
        context_max_chars=settings.phase3_context_max_chars,
        checkpointer=checkpointer,
        session_manager=build_session_manager(settings),
    )


def build_phase4(settings: Settings, *, checkpointer=None):
    providers = build_provider_registry(settings, include_generation=True)
    pipeline = build_phase12(settings, providers=providers)
    return create_phase4_orchestrator(
        pipeline,
        _build_generation_service(settings, providers),
        default_top_k=settings.phase3_default_top_k,
        context_max_chars=settings.phase3_context_max_chars,
        checkpointer=checkpointer,
        session_manager=build_session_manager(settings),
    )


def build_phase5(settings: Settings, *, checkpointer=None):
    providers = build_provider_registry(settings, include_generation=True)
    pipeline = build_phase12(settings, providers=providers)
    return create_phase5_orchestrator(
        pipeline,
        _build_generation_service(settings, providers),
        _build_verification_service(settings, providers),
        default_top_k=settings.phase3_default_top_k,
        context_max_chars=settings.phase3_context_max_chars,
        checkpointer=checkpointer,
        session_manager=build_session_manager(settings),
    )


def build_phase6(settings: Settings, *, checkpointer=None):
    providers = build_provider_registry(settings, include_generation=True, include_image_generation=True)
    pipeline = build_phase12(settings, providers=providers)
    return create_phase6_orchestrator(
        pipeline,
        _build_generation_service(settings, providers),
        _build_verification_service(settings, providers),
        build_artifact_service(settings, providers),
        default_top_k=settings.phase3_default_top_k,
        context_max_chars=settings.phase3_context_max_chars,
        checkpointer=checkpointer,
        session_manager=build_session_manager(settings),
    )
