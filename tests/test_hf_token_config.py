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
    "SESSION_STORE_BACKEND": "memory",
}


def _set_env(monkeypatch, values=None):
    prefixes = (
        "HF_", "HARRIER_", "LLM_", "SIGLIP_", "VLM_", "CHROMA_",
        "RERANKER_", "VERIFIER_", "SESSION_", "IMAGE_GEN_",
    )
    for key in list(os.environ):
        if key.startswith(prefixes):
            monkeypatch.delenv(key, raising=False)
    for key, value in (values or BASE_ENV).items():
        monkeypatch.setenv(key, value)


def test_single_hf_token_configures_hf_model_layer(monkeypatch):
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

    # Verification reuses the primary HF-hosted Qwen provider by default.
    assert settings.verifier_api_url is None
    assert settings.verifier_api_key is None

    # Visual providers have real Hugging Face defaults; no endpoint URLs are required.
    assert settings.siglip_api_key == "hf_test_token"
    assert settings.siglip_api_url is None
    assert settings.siglip_model == "google/siglip-so400m-patch14-384"
    assert settings.vlm_api_key == "hf_test_token"
    assert settings.vlm_api_url == "https://router.huggingface.co/v1/chat/completions"
    assert settings.vlm_model == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert settings.vlm_fallback_models == ("zai-org/GLM-4.5V",)
    assert settings.vlm_api_style == "openai"


def test_hf_reranker_endpoint_reuses_hf_token(monkeypatch):
    values = dict(BASE_ENV)
    values["RERANKER_API_URL"] = "https://reranker.endpoints.huggingface.cloud"
    _set_env(monkeypatch, values)
    settings = Settings.from_env()

    assert settings.reranker_api_key == "hf_test_token"
    assert settings.reranker_api_style == "hf_tei"
    assert settings.reranker_model == "BAAI/bge-reranker-v2-m3"


def test_provider_specific_key_still_overrides_hf_token_for_custom_deployments(monkeypatch):
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


def test_placeholder_vlm_endpoint_is_ignored_and_hf_router_is_used(monkeypatch):
    values = dict(BASE_ENV)
    values["VLM_API_URL"] = "https://your-vlm-endpoint.endpoints.huggingface.cloud"
    _set_env(monkeypatch, values)
    settings = Settings.from_env()

    assert settings.vlm_api_url == "https://router.huggingface.co/v1/chat/completions"
    assert settings.vlm_api_key == "hf_test_token"
