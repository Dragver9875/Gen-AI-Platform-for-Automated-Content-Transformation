from __future__ import annotations

import os
from dataclasses import dataclass


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


@dataclass(frozen=True)
class Settings:
    # Chroma Cloud
    chroma_api_key: str
    chroma_tenant: str
    chroma_database: str
    chroma_collection: str

    # Harrier hosted endpoint
    harrier_api_url: str
    harrier_api_key: str
    harrier_api_style: str
    harrier_query_instruction: str
    harrier_batch_size: int

    # Optional hosted reranker endpoint. RRF is used when omitted.
    reranker_api_url: str | None
    reranker_api_key: str | None

    # Docling managed/remote service
    docling_api_url: str
    docling_api_key: str | None
    docling_timeout_s: float

    # Hosted SigLIP router endpoint
    siglip_api_url: str
    siglip_api_key: str

    # Hosted VLM endpoint
    vlm_api_url: str
    vlm_api_key: str

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

    # Phase 3 orchestration
    phase3_default_top_k: int
    phase3_context_max_chars: int

    # Phase 4 hosted SLM + CRR generation
    llm_api_url: str | None
    llm_api_key: str | None
    llm_api_style: str
    llm_model: str | None
    llm_response_mode: str
    phase4_temperature: float
    phase4_max_tokens: int
    phase4_group_context_max_chars: int
    phase4_digest_max_tokens: int

    # HTTP
    http_timeout_s: float
    http_retries: int

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            chroma_api_key=_env("CHROMA_API_KEY", required=True),  # type: ignore[arg-type]
            chroma_tenant=_env("CHROMA_TENANT", required=True),  # type: ignore[arg-type]
            chroma_database=_env("CHROMA_DATABASE", required=True),  # type: ignore[arg-type]
            chroma_collection=_env("CHROMA_COLLECTION", "document_chunks") or "document_chunks",
            harrier_api_url=_env("HARRIER_API_URL", required=True),  # type: ignore[arg-type]
            harrier_api_key=_env("HARRIER_API_KEY", required=True),  # type: ignore[arg-type]
            harrier_api_style=(_env("HARRIER_API_STYLE", "hf") or "hf").lower(),
            harrier_query_instruction=_env(
                "HARRIER_QUERY_INSTRUCTION",
                "Retrieve passages from the supplied documents that answer the user's query:",
            ) or "Retrieve passages from the supplied documents that answer the user's query:",
            harrier_batch_size=_env_int("HARRIER_BATCH_SIZE", 64),
            reranker_api_url=_env("RERANKER_API_URL"),
            reranker_api_key=_env("RERANKER_API_KEY"),
            docling_api_url=_env("DOCLING_API_URL", required=True),  # type: ignore[arg-type]
            docling_api_key=_env("DOCLING_API_KEY"),
            docling_timeout_s=_env_float("DOCLING_TIMEOUT_S", 180.0),
            siglip_api_url=_env("SIGLIP_API_URL", required=True),  # type: ignore[arg-type]
            siglip_api_key=_env("SIGLIP_API_KEY", required=True),  # type: ignore[arg-type]
            vlm_api_url=_env("VLM_API_URL", required=True),  # type: ignore[arg-type]
            vlm_api_key=_env("VLM_API_KEY", required=True),  # type: ignore[arg-type]
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
            phase3_default_top_k=_env_int("PHASE3_DEFAULT_TOP_K", 5),
            phase3_context_max_chars=_env_int("PHASE3_CONTEXT_MAX_CHARS", 60000),
            llm_api_url=_env("LLM_API_URL"),
            llm_api_key=_env("LLM_API_KEY"),
            llm_api_style=(_env("LLM_API_STYLE", "openai") or "openai").lower(),
            llm_model=_env("LLM_MODEL"),
            llm_response_mode=(_env("LLM_RESPONSE_MODE", "json_object") or "json_object").lower(),
            phase4_temperature=_env_float("PHASE4_TEMPERATURE", 0.1),
            phase4_max_tokens=_env_int("PHASE4_MAX_TOKENS", 4096),
            phase4_group_context_max_chars=_env_int("PHASE4_GROUP_CONTEXT_MAX_CHARS", 18000),
            phase4_digest_max_tokens=_env_int("PHASE4_DIGEST_MAX_TOKENS", 2048),
            http_timeout_s=_env_float("HTTP_TIMEOUT_S", 90.0),
            http_retries=_env_int("HTTP_RETRIES", 2),
        )
