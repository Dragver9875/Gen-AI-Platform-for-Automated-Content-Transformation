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

    # Ingestion routing
    pdf_native_text_chars: int
    pdf_image_coverage_threshold: float
    pdf_visual_fallback_enabled: bool
    pdf_visual_fallback_max_pages: int

    # Chunking
    chunk_target_chars: int
    chunk_overlap_chars: int

    # Phase 3 orchestration
    phase3_default_top_k: int
    phase3_context_max_chars: int

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
            pdf_native_text_chars=_env_int("PDF_NATIVE_TEXT_CHARS", 80),
            pdf_image_coverage_threshold=_env_float("PDF_IMAGE_COVERAGE_THRESHOLD", 0.72),
            pdf_visual_fallback_enabled=_env_bool("PDF_VISUAL_FALLBACK_ENABLED", True),
            pdf_visual_fallback_max_pages=_env_int("PDF_VISUAL_FALLBACK_MAX_PAGES", 12),
            chunk_target_chars=_env_int("CHUNK_TARGET_CHARS", 3200),
            chunk_overlap_chars=_env_int("CHUNK_OVERLAP_CHARS", 450),
            phase3_default_top_k=_env_int("PHASE3_DEFAULT_TOP_K", 5),
            phase3_context_max_chars=_env_int("PHASE3_CONTEXT_MAX_CHARS", 60000),
            http_timeout_s=_env_float("HTTP_TIMEOUT_S", 90.0),
            http_retries=_env_int("HTTP_RETRIES", 2),
        )
