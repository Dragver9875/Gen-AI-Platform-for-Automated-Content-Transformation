from __future__ import annotations

import os

import pytest

from app.config import ConfigurationError, Settings
from providers.harrier import HarrierEmbeddingProvider


BASE_ENV = {
    "HF_TOKEN": "hf_test_token",
    "CHROMA_API_KEY": "chroma",
    "CHROMA_TENANT": "tenant",
    "CHROMA_DATABASE": "db",
    "DOCLING_API_URL": "https://docling.example",
    "SIGLIP_API_URL": "https://siglip.endpoints.huggingface.cloud",
    "VLM_API_URL": "https://vlm.endpoints.huggingface.cloud",
    "SESSION_STORE_BACKEND": "memory",
}


def _set_env(monkeypatch, values=None):
    for key in list(os.environ):
        if key.startswith(("HF_", "HARRIER_", "LLM_", "SIGLIP_", "VLM_", "CHROMA_", "DOCLING_", "SESSION_")):
            monkeypatch.delenv(key, raising=False)
    for key, value in (values or BASE_ENV).items():
        monkeypatch.setenv(key, value)


def test_single_hf_token_configures_harrier_and_default_qwen_router(monkeypatch):
    _set_env(monkeypatch)
    settings = Settings.from_env()

    assert settings.harrier_api_key == "hf_test_token"
    assert settings.harrier_api_url == (
        "https://router.huggingface.co/hf-inference/models/"
        "microsoft/harrier-oss-v1-0.6b"
    )
    assert settings.harrier_prompt_name == "web_search_query"
    assert settings.harrier_normalize is True

    assert settings.llm_api_key == "hf_test_token"
    assert settings.llm_api_url == "https://router.huggingface.co/v1/chat/completions"

    # HF-hosted custom endpoints safely reuse the same token.
    assert settings.siglip_api_key == "hf_test_token"
    assert settings.vlm_api_key == "hf_test_token"


def test_provider_specific_key_overrides_hf_token(monkeypatch):
    values = dict(BASE_ENV)
    values["HARRIER_API_KEY"] = "dedicated_harrier_key"
    _set_env(monkeypatch, values)

    settings = Settings.from_env()
    assert settings.harrier_api_key == "dedicated_harrier_key"


def test_hf_token_is_not_forwarded_to_non_hf_custom_endpoint(monkeypatch):
    values = dict(BASE_ENV)
    values["VLM_API_URL"] = "https://vision.example.com/infer"
    _set_env(monkeypatch, values)

    with pytest.raises(ConfigurationError, match="VLM_API_KEY"):
        Settings.from_env()


def test_harrier_hf_payload_uses_query_prompt_and_no_document_prompt():
    provider = HarrierEmbeddingProvider(
        "https://router.huggingface.co/hf-inference/models/microsoft/harrier-oss-v1-0.6b",
        "hf_test",
        prompt_name="web_search_query",
        normalize=True,
    )

    query_payload = provider._payload(["flood mitigation"], prompt_name=provider.prompt_name)
    assert query_payload == {
        "inputs": ["flood mitigation"],
        "normalize": True,
        "prompt_name": "web_search_query",
    }

    document_payload = provider._payload(["document text"], prompt_name=None)
    assert document_payload == {
        "inputs": ["document text"],
        "normalize": True,
    }
