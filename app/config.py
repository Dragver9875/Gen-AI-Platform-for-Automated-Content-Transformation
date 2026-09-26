from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


class ConfigurationError(RuntimeError):
    pass


def _secret_file_value(name: str) -> str | None:
    """Read a Render Secret File when the equivalent environment variable is absent."""
    secret_dir = Path(os.getenv("RENDER_SECRET_DIR", "/etc/secrets"))
    path = secret_dir / name
    try:
        if path.is_file():
            value = path.read_text(encoding="utf-8").strip()
            return value or None
    except (OSError, UnicodeError):
        pass
    return None


def _env(name: str, default: str | None = None, *, required: bool = False) -> str | None:
    raw = os.getenv(name)
    value = raw.strip() if raw is not None and raw.strip() else _secret_file_value(name)
    if value is None:
        value = default
    if required and not value:
        raise ConfigurationError(
            f"Missing required configuration: {name}. "
            f"Set it as an environment variable or Render Secret File /etc/secrets/{name}."
        )
    return value


def _first_env(*names: str) -> str | None:
    """Return the first non-empty value from environment variables or Render Secret Files."""
    for name in names:
        value = _env(name)
        if value:
            return value
    return None


def _env_bool(name: str, default: bool = False) -> bool:
    value = _env(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = _env(name)
    return int(value) if value else default


def _env_float(name: str, default: float) -> float:
    value = _env(name)
    return float(value) if value else default


def _is_huggingface_url(url: str | None) -> bool:
    if not url:
        return False
    host = (urlparse(url).hostname or "").lower()
    return host == "huggingface.co" or host.endswith(".huggingface.co") or host.endswith(".huggingface.cloud")


def _clean_endpoint(value: str | None) -> str | None:
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
    # Shared credential for all Hugging Face-hosted ML services.
    hf_token: str | None

    # Chroma Cloud
    chroma_api_key: str
    chroma_tenant: str
    chroma_database: str
    chroma_collection: str

    # Harrier embeddings
    harrier_api_url: str
    harrier_api_key: str
    harrier_api_style: str
    harrier_model: str
    harrier_prompt_name: str | None
    harrier_normalize: bool
    harrier_query_instruction: str
    harrier_batch_size: int

    # Optional HF-compatible reranker endpoint. RRF is used when omitted.
    reranker_api_url: str | None
    reranker_api_key: str | None
    reranker_api_style: str
    reranker_model: str

    # Single shared multimodal encoder: Qwen2.5-VL.
    multimodal_api_url: str
    multimodal_api_key: str
    multimodal_api_style: str
    multimodal_model: str
    multimodal_max_tokens: int
    multimodal_render_dpi: int
    multimodal_max_pdf_pages: int
    multimodal_max_pptx_slides: int
    multimodal_document_prompt: str
    multimodal_image_prompt: str

    # User sessions
    session_store_backend: str
    session_database_url: str | None
    session_table: str

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

    # Main semantic decoder: gpt-oss-20b. Reused for generation, verification and repair.
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

    # Creative image decoder. Defaults to FLUX.1-schnell via HF InferenceClient.
    image_gen_api_url: str | None
    image_gen_api_key: str | None
    image_gen_api_style: str
    image_gen_model: str | None
    image_gen_provider: str

    # HTTP
    http_timeout_s: float
    http_retries: int

    @classmethod
    def from_env(cls) -> "Settings":
        hf_token = _first_env("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HUGGING_FACE_HUB_TOKEN")

        harrier_model = _env("HARRIER_MODEL", "microsoft/harrier-oss-v1-0.6b") or "microsoft/harrier-oss-v1-0.6b"
        harrier_api_url = _clean_endpoint(_env("HARRIER_API_URL")) or (
            f"https://router.huggingface.co/hf-inference/models/{harrier_model}"
        )
        harrier_api_key = _provider_key(
            "HARRIER_API_KEY", api_url=harrier_api_url, hf_token=hf_token, required=True
        )

        # New names are QWEN_VL_*. VLM_* is read only as a compatibility alias
        # so old .env files do not break during migration.
        multimodal_model = (
            _env("QWEN_VL_MODEL")
            or _env("VLM_MODEL")
            or "Qwen/Qwen2.5-VL-3B-Instruct:featherless-ai"
        )
        multimodal_api_url = (
            _clean_endpoint(_env("QWEN_VL_API_URL"))
            or _clean_endpoint(_env("VLM_API_URL"))
            or "https://router.huggingface.co/v1/chat/completions"
        )
        multimodal_api_key = (
            _env("QWEN_VL_API_KEY")
            or _provider_key("VLM_API_KEY", api_url=multimodal_api_url, hf_token=hf_token, required=True)
        )
        multimodal_api_style = (
            _env("QWEN_VL_API_STYLE") or _env("VLM_API_STYLE") or "openai"
        ).lower()

        llm_api_url = _clean_endpoint(_env("LLM_API_URL"))
        if not llm_api_url and hf_token:
            llm_api_url = "https://router.huggingface.co/v1/chat/completions"
        llm_model = _env("LLM_MODEL", "openai/gpt-oss-20b:fastest") or "openai/gpt-oss-20b:fastest"
        if llm_model == "Qwen/Qwen3-30B-A3B-Instruct-2507" and _is_huggingface_url(llm_api_url):
            llm_model = "openai/gpt-oss-20b:fastest"
        llm_api_key = _provider_key("LLM_API_KEY", api_url=llm_api_url, hf_token=hf_token, required=False)

        reranker_api_url = _clean_endpoint(_env("RERANKER_API_URL"))
        reranker_api_key = _provider_key(
            "RERANKER_API_KEY", api_url=reranker_api_url, hf_token=hf_token, required=False
        )

        # Creative image generation defaults to the HF client/provider broker,
        # so no endpoint URL or separate credential is required.
        image_gen_api_url = _clean_endpoint(_env("IMAGE_GEN_API_URL"))
        image_gen_api_style = (_env("IMAGE_GEN_API_STYLE", "hf_hub") or "hf_hub").lower()
        if image_gen_api_style == "hf_hub":
            image_gen_api_key = _env("IMAGE_GEN_API_KEY") or hf_token
        else:
            image_gen_api_key = _provider_key(
                "IMAGE_GEN_API_KEY", api_url=image_gen_api_url, hf_token=hf_token, required=False
            )

        return cls(
            hf_token=hf_token,
            chroma_api_key=(
                _first_env("CHROMA_API_KEY", "CHROMA_CLOUD_API_KEY")
                or (_ for _ in ()).throw(ConfigurationError("Missing required environment variable: CHROMA_API_KEY"))
            ),
            chroma_tenant=(
                _first_env("CHROMA_TENANT", "CHROMA_CLOUD_TENANT")
                or (_ for _ in ()).throw(ConfigurationError("Missing required environment variable: CHROMA_TENANT"))
            ),
            chroma_database=(
                _first_env("CHROMA_DATABASE", "CHROMA_CLOUD_DATABASE")
                or (_ for _ in ()).throw(ConfigurationError("Missing required environment variable: CHROMA_DATABASE"))
            ),
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
            multimodal_api_url=multimodal_api_url,
            multimodal_api_key=multimodal_api_key or "",
            multimodal_api_style=multimodal_api_style,
            multimodal_model=multimodal_model,
            multimodal_max_tokens=_env_int("QWEN_VL_MAX_TOKENS", 2400),
            multimodal_render_dpi=_env_int("MULTIMODAL_RENDER_DPI", 144),
            multimodal_max_pdf_pages=_env_int("MULTIMODAL_MAX_PDF_PAGES", 40),
            multimodal_max_pptx_slides=_env_int("MULTIMODAL_MAX_PPTX_SLIDES", 40),
            multimodal_document_prompt=_env(
                "MULTIMODAL_DOCUMENT_PROMPT",
                "You are the shared multimodal document encoder for a retrieval system. Read this page or slide faithfully and return retrieval-ready Markdown. Preserve headings, paragraphs, lists, tables, equations, labels, numbers, chart values and trends, diagram relationships, captions, and reading order. Describe meaningful non-text visuals. Do not summarize, omit facts, or invent content.",
            ) or "",
            multimodal_image_prompt=_env(
                "MULTIMODAL_IMAGE_PROMPT",
                "You are the shared multimodal encoder for a retrieval system. Understand this image faithfully and return retrieval-ready Markdown. Preserve all visible text, entities, numbers, labels, spatial or causal relationships, chart or diagram semantics, and other information needed to answer questions about the image. If the image is a photographed or scanned document page, transcribe it rather than merely describing it. Do not invent details.",
            ) or "",
            session_store_backend=(_env("SESSION_STORE_BACKEND", "memory") or "memory").lower(),
            session_database_url=_env("SESSION_DATABASE_URL"),
            session_table=_env("SESSION_TABLE", "user_sessions") or "user_sessions",
            chunk_target_chars=_env_int("CHUNK_TARGET_CHARS", 3200),
            chunk_overlap_chars=_env_int("CHUNK_OVERLAP_CHARS", 450),
            retrieval_strategy=(_env("RETRIEVAL_STRATEGY", "hybrid") or "hybrid").lower(),
            retrieval_bm25_k=_env_int("RETRIEVAL_BM25_K", 15),
            retrieval_vector_k=_env_int("RETRIEVAL_VECTOR_K", 15),
            retrieval_rrf_k=_env_int("RETRIEVAL_RRF_K", 60),
            retrieval_use_reranker=_env_bool("RETRIEVAL_USE_RERANKER", False),
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
            image_gen_api_style=image_gen_api_style,
            image_gen_model=_env("IMAGE_GEN_MODEL", "black-forest-labs/FLUX.1-schnell") or "black-forest-labs/FLUX.1-schnell",
            image_gen_provider=_env("IMAGE_GEN_PROVIDER", "auto") or "auto",
            http_timeout_s=_env_float("HTTP_TIMEOUT_S", 120.0),
            http_retries=_env_int("HTTP_RETRIES", 2),
        )
