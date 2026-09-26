from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlparse


class ConfigurationError(RuntimeError):
    pass


def _env(name: str, default: str | None = None, *, required: bool = False) -> str | None:
    value = os.getenv(name, default)
    if required and not value:
        raise ConfigurationError(f"Missing required environment variable: {name}")
    return value


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


def _is_huggingface_url(url: str | None) -> bool:
    if not url:
        return False
    host = (urlparse(url).hostname or "").lower()
    return (
        host == "huggingface.co"
        or host.endswith(".huggingface.co")
        or host.endswith(".huggingface.cloud")
    )




def _clean_endpoint(value: str | None) -> str | None:
    """Treat common template/example endpoint values as unset.

    Older repository revisions shipped examples such as
    ``https://your-vlm-endpoint.endpoints.huggingface.cloud``. Calling those
    values literally produces a confusing provider 404 instead of falling back
    to the shared HF router.
    """
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    host = (urlparse(value).hostname or "").lower()
    lower = value.lower()
    if (
        host.startswith(("your-", "example-", "placeholder-"))
        or host in {"example.com", "www.example.com"}
        or "your-vlm-endpoint" in lower
        or "your-endpoint" in lower
        or "replace-me" in lower
        or "changeme" in lower
    ):
        return None
    return value

def _provider_key(
    env_name: str,
    *,
    api_url: str | None,
    hf_token: str | None,
    required: bool = False,
) -> str | None:
    """Resolve an endpoint credential safely.

    A shared HF_TOKEN is only reused for Hugging Face-owned router/endpoint URLs.
    This avoids accidentally forwarding a Hugging Face token to an unrelated
    third-party URL when a provider-specific key is omitted.
    """
    explicit = _env(env_name)
    if explicit:
        return explicit
    if hf_token and _is_huggingface_url(api_url):
        return hf_token
    if required:
        raise ConfigurationError(
            f"Missing required environment variable: {env_name}. "
            "For Hugging Face endpoints you may set HF_TOKEN instead."
        )
    return None


@dataclass(frozen=True)
class Settings:
    # Shared Hugging Face credential. Used only for Hugging Face-owned URLs.
    hf_token: str | None

    # Chroma Cloud
    chroma_api_key: str
    chroma_tenant: str
    chroma_database: str
    chroma_collection: str

    # Harrier hosted/serverless endpoint
    harrier_api_url: str
    harrier_api_key: str
    harrier_api_style: str
    harrier_model: str
    harrier_prompt_name: str | None
    harrier_normalize: bool
    harrier_query_instruction: str
    harrier_batch_size: int

    # Optional hosted reranker endpoint. RRF is used when omitted.
    reranker_api_url: str | None
    reranker_api_key: str | None
    reranker_api_style: str
    reranker_model: str

    # Optional SigLIP routing. Disabled by default because serverless availability is inconsistent.
    siglip_enabled: bool
    siglip_api_url: str | None
    siglip_api_key: str
    siglip_model: str

    # Hosted VLM. Defaults to Hugging Face OpenAI-compatible multimodal router.
    vlm_api_url: str
    vlm_api_key: str
    vlm_model: str
    vlm_fallback_models: tuple[str, ...]
    vlm_api_style: str

    # User sessions
    session_store_backend: str
    session_database_url: str | None
    session_table: str

    # Ingestion routing
    pdf_native_text_chars: int
    pdf_image_coverage_threshold: float
    pdf_visual_fallback_enabled: bool
    pdf_visual_fallback_max_pages: int
    siglip_document_labels: tuple[str, ...]
    pdf_visual_prompt: str

    # Chunking
    chunk_target_chars: int
    chunk_overlap_chars: int

    # Retrieval strategy
    retrieval_strategy: str
    retrieval_bm25_k: int
    retrieval_vector_k: int
    retrieval_rrf_k: int
    retrieval_use_reranker: bool

    # Phase 3 orchestration
    phase3_default_top_k: int
    phase3_context_max_chars: int

    # Phase 4 hosted open-source LLM + CRR generation
    llm_api_url: str | None
    llm_api_key: str | None
    llm_api_style: str
    llm_model: str | None
    llm_response_mode: str
    phase4_temperature: float
    phase4_max_tokens: int
    phase4_group_context_max_chars: int
    phase4_digest_max_tokens: int

    # Phase 5 factuality verification + bounded repair
    verifier_api_url: str | None
    verifier_api_key: str | None
    verifier_api_style: str
    verifier_model: str | None
    verifier_response_mode: str
    phase5_verification_temperature: float
    phase5_verification_max_tokens: int
    phase5_repair_temperature: float
    phase5_repair_max_tokens: int
    phase5_max_repair_attempts: int
    phase5_min_faithfulness_score: float
    phase5_max_claims_per_call: int
    phase5_evidence_chars_per_claim: int
    phase5_default_verification_profile: str
    phase5_verification_profiles_json: str | None

    # Phase 6 artifact generation
    phase6_output_dir: str
    phase6_artifact_specs_path: str
    phase6_typst_binary: str
    phase6_retain_typst_source: bool
    phase6_fail_fast: bool
    phase6_temperature: float
    phase6_max_tokens: int

    # Hosted image generation (e.g. FLUX endpoint)
    image_gen_api_url: str | None
    image_gen_api_key: str | None
    image_gen_api_style: str
    image_gen_model: str | None

    # HTTP
    http_timeout_s: float
    http_retries: int

    @classmethod
    def from_env(cls) -> "Settings":
        hf_token = _env("HF_TOKEN")

        harrier_model = _env("HARRIER_MODEL", "microsoft/harrier-oss-v1-0.6b") or "microsoft/harrier-oss-v1-0.6b"
        harrier_api_url = _clean_endpoint(_env("HARRIER_API_URL")) or (
            f"https://router.huggingface.co/hf-inference/models/{harrier_model}"
        )
        harrier_api_key = _provider_key(
            "HARRIER_API_KEY",
            api_url=harrier_api_url,
            hf_token=hf_token,
            required=True,
        )

        # All hosted ML services reuse HF_TOKEN on Hugging Face infrastructure.
        # Native PDF/PPTX parsing is deterministic and does not require a document-model endpoint.

        siglip_enabled = _env_bool("SIGLIP_ENABLED", False)
        siglip_model = _env("SIGLIP_MODEL", "google/siglip-so400m-patch14-384") or "google/siglip-so400m-patch14-384"
        siglip_api_url = _clean_endpoint(_env("SIGLIP_API_URL"))
        if not siglip_enabled:
            siglip_api_key = _env("SIGLIP_API_KEY") or ""
        elif siglip_api_url:
            siglip_api_key = _provider_key(
                "SIGLIP_API_KEY",
                api_url=siglip_api_url,
                hf_token=hf_token,
                required=True,
            )
        else:
            # The default SigLIP path uses huggingface_hub.InferenceClient directly.
            # No endpoint URL is required; the Hub selects an available inference provider.
            siglip_api_key = _env("SIGLIP_API_KEY") or hf_token
            if not siglip_api_key:
                raise ConfigurationError("HF_TOKEN is required for default SigLIP inference")

        vlm_model = _env("VLM_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct") or "Qwen/Qwen2.5-VL-3B-Instruct"
        vlm_fallback_models = tuple(
            x.strip()
            for x in (_env("VLM_FALLBACK_MODELS", "zai-org/GLM-4.5V") or "").split(",")
            if x.strip() and x.strip() != vlm_model
        )
        vlm_api_url = _clean_endpoint(_env("VLM_API_URL")) or "https://router.huggingface.co/v1/chat/completions"
        vlm_api_key = _provider_key(
            "VLM_API_KEY",
            api_url=vlm_api_url,
            hf_token=hf_token,
            required=True,
        )
        vlm_api_style = (_env("VLM_API_STYLE", "openai") or "openai").lower()

        llm_api_url = _clean_endpoint(_env("LLM_API_URL"))
        if not llm_api_url and hf_token:
            # Hugging Face Inference Providers expose an OpenAI-compatible chat route.
            llm_api_url = "https://router.huggingface.co/v1/chat/completions"

        # Compatibility migration: an earlier repository revision defaulted to a
        # Qwen3 checkpoint that is not currently router-served by HF Inference
        # Providers. Preserve explicit custom endpoints, but transparently migrate
        # that legacy model when using the shared HF router.
        llm_model = _env("LLM_MODEL", "openai/gpt-oss-20b:fastest") or "openai/gpt-oss-20b:fastest"
        if (
            llm_model == "Qwen/Qwen3-30B-A3B-Instruct-2507"
            and _is_huggingface_url(llm_api_url)
        ):
            llm_model = "openai/gpt-oss-20b:fastest"

        llm_api_key = _provider_key(
            "LLM_API_KEY",
            api_url=llm_api_url,
            hf_token=hf_token,
            required=False,
        )

        reranker_api_url = _clean_endpoint(_env("RERANKER_API_URL"))
        reranker_api_key = _provider_key(
            "RERANKER_API_KEY",
            api_url=reranker_api_url,
            hf_token=hf_token,
            required=False,
        )

        verifier_api_url = _clean_endpoint(_env("VERIFIER_API_URL"))
        verifier_api_key = _provider_key(
            "VERIFIER_API_KEY",
            api_url=verifier_api_url,
            hf_token=hf_token,
            required=False,
        )

        image_gen_api_url = _clean_endpoint(_env("IMAGE_GEN_API_URL"))
        image_gen_api_key = _provider_key(
            "IMAGE_GEN_API_KEY",
            api_url=image_gen_api_url,
            hf_token=hf_token,
            required=False,
        )

        return cls(
            hf_token=hf_token,
            chroma_api_key=_env("CHROMA_API_KEY", required=True),  # type: ignore[arg-type]
            chroma_tenant=_env("CHROMA_TENANT", required=True),  # type: ignore[arg-type]
            chroma_database=_env("CHROMA_DATABASE", required=True),  # type: ignore[arg-type]
            chroma_collection=_env("CHROMA_COLLECTION", "document_chunks") or "document_chunks",
            harrier_api_url=harrier_api_url,
            harrier_api_key=harrier_api_key or "",
            harrier_api_style=(_env("HARRIER_API_STYLE", "hf") or "hf").lower(),
            harrier_model=harrier_model,
            harrier_prompt_name=_env("HARRIER_PROMPT_NAME", "web_search_query"),
            harrier_normalize=_env_bool("HARRIER_NORMALIZE", True),
            harrier_query_instruction=_env(
                "HARRIER_QUERY_INSTRUCTION",
                "Given a web search query, retrieve relevant passages that answer the query",
            ) or "Given a web search query, retrieve relevant passages that answer the query",
            harrier_batch_size=_env_int("HARRIER_BATCH_SIZE", 64),
            reranker_api_url=reranker_api_url,
            reranker_api_key=reranker_api_key,
            reranker_api_style=(_env("RERANKER_API_STYLE", "hf_tei") or "hf_tei").lower(),
            reranker_model=_env("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3") or "BAAI/bge-reranker-v2-m3",
            siglip_enabled=siglip_enabled,
            siglip_api_url=siglip_api_url,
            siglip_api_key=siglip_api_key or "",
            siglip_model=siglip_model,
            vlm_api_url=vlm_api_url,
            vlm_api_key=vlm_api_key or "",
            vlm_model=vlm_model,
            vlm_fallback_models=vlm_fallback_models,
            vlm_api_style=vlm_api_style,
            session_store_backend=(_env("SESSION_STORE_BACKEND", "memory") or "memory").lower(),
            session_database_url=_env("SESSION_DATABASE_URL"),
            session_table=_env("SESSION_TABLE", "user_sessions") or "user_sessions",
            pdf_native_text_chars=_env_int("PDF_NATIVE_TEXT_CHARS", 80),
            pdf_image_coverage_threshold=_env_float("PDF_IMAGE_COVERAGE_THRESHOLD", 0.72),
            pdf_visual_fallback_enabled=_env_bool("PDF_VISUAL_FALLBACK_ENABLED", True),
            pdf_visual_fallback_max_pages=_env_int("PDF_VISUAL_FALLBACK_MAX_PAGES", 12),
            siglip_document_labels=tuple(x.strip().lower() for x in (_env("SIGLIP_DOCUMENT_LABELS", "document page,screenshot") or "").split(",") if x.strip()),
            pdf_visual_prompt=_env("PDF_VISUAL_PROMPT", "Describe this PDF page faithfully for retrieval. Preserve all visible text, labels, numbers, chart trends, diagram relationships, and important visual content. If it is primarily a scanned text page, transcribe the meaningful content.") or "",
            chunk_target_chars=_env_int("CHUNK_TARGET_CHARS", 3200),
            chunk_overlap_chars=_env_int("CHUNK_OVERLAP_CHARS", 450),
            retrieval_strategy=(_env("RETRIEVAL_STRATEGY", "hybrid") or "hybrid").lower(),
            retrieval_bm25_k=_env_int("RETRIEVAL_BM25_K", 15),
            retrieval_vector_k=_env_int("RETRIEVAL_VECTOR_K", 15),
            retrieval_rrf_k=_env_int("RETRIEVAL_RRF_K", 60),
            retrieval_use_reranker=_env_bool("RETRIEVAL_USE_RERANKER", True),
            phase3_default_top_k=_env_int("PHASE3_DEFAULT_TOP_K", 5),
            phase3_context_max_chars=_env_int("PHASE3_CONTEXT_MAX_CHARS", 60000),
            llm_api_url=llm_api_url,
            llm_api_key=llm_api_key,
            llm_api_style=(_env("LLM_API_STYLE", "openai") or "openai").lower(),
            llm_model=llm_model,
            llm_response_mode=(_env("LLM_RESPONSE_MODE", "json_schema") or "json_schema").lower(),
            phase4_temperature=_env_float("PHASE4_TEMPERATURE", 0.1),
            phase4_max_tokens=_env_int("PHASE4_MAX_TOKENS", 4096),
            phase4_group_context_max_chars=_env_int("PHASE4_GROUP_CONTEXT_MAX_CHARS", 18000),
            phase4_digest_max_tokens=_env_int("PHASE4_DIGEST_MAX_TOKENS", 2048),
            verifier_api_url=verifier_api_url,
            verifier_api_key=verifier_api_key,
            verifier_api_style=(_env("VERIFIER_API_STYLE", "openai") or "openai").lower(),
            verifier_model=_env("VERIFIER_MODEL"),
            verifier_response_mode=(_env("VERIFIER_RESPONSE_MODE", "json_object") or "json_object").lower(),
            phase5_verification_temperature=_env_float("PHASE5_VERIFICATION_TEMPERATURE", 0.0),
            phase5_verification_max_tokens=_env_int("PHASE5_VERIFICATION_MAX_TOKENS", 4096),
            phase5_repair_temperature=_env_float("PHASE5_REPAIR_TEMPERATURE", 0.05),
            phase5_repair_max_tokens=_env_int("PHASE5_REPAIR_MAX_TOKENS", 4096),
            phase5_max_repair_attempts=_env_int("PHASE5_MAX_REPAIR_ATTEMPTS", 2),
            phase5_min_faithfulness_score=_env_float("PHASE5_MIN_FAITHFULNESS_SCORE", 1.0),
            phase5_max_claims_per_call=_env_int("PHASE5_MAX_CLAIMS_PER_CALL", 12),
            phase5_evidence_chars_per_claim=_env_int("PHASE5_EVIDENCE_CHARS_PER_CLAIM", 8000),
            phase5_default_verification_profile=(_env("PHASE5_DEFAULT_VERIFICATION_PROFILE", "strict") or "strict").lower(),
            phase5_verification_profiles_json=_env("PHASE5_VERIFICATION_PROFILES_JSON"),
            phase6_output_dir=_env("PHASE6_OUTPUT_DIR", "runtime_artifacts") or "runtime_artifacts",
            phase6_artifact_specs_path=_env("PHASE6_ARTIFACT_SPECS_PATH", "artifacts/specs/default.json") or "artifacts/specs/default.json",
            phase6_typst_binary=_env("PHASE6_TYPST_BINARY", "typst") or "typst",
            phase6_retain_typst_source=_env_bool("PHASE6_RETAIN_TYPST_SOURCE", True),
            phase6_fail_fast=_env_bool("PHASE6_FAIL_FAST", False),
            phase6_temperature=_env_float("PHASE6_TEMPERATURE", 0.1),
            phase6_max_tokens=_env_int("PHASE6_MAX_TOKENS", 4096),
            image_gen_api_url=image_gen_api_url,
            image_gen_api_key=image_gen_api_key,
            image_gen_api_style=(_env("IMAGE_GEN_API_STYLE", "hf") or "hf").lower(),
            image_gen_model=_env("IMAGE_GEN_MODEL"),
            http_timeout_s=_env_float("HTTP_TIMEOUT_S", 90.0),
            http_retries=_env_int("HTTP_RETRIES", 2),
        )
