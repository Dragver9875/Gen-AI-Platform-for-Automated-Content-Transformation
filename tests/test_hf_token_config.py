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
        "HF_", "HARRIER_", "LLM_", "VLM_", "QWEN_VL_", "MULTIMODAL_", "CHROMA_",
        "RERANKER_", "VERIFIER_", "SIGLIP_", "SESSION_", "IMAGE_GEN_",
    )
    for key in list(os.environ):
        if key.startswith(prefixes):
            monkeypatch.delenv(key, raising=False)
    for key, value in (values or BASE_ENV).items():
        monkeypatch.setenv(key, value)


def test_single_hf_token_configures_entire_model_layer(monkeypatch):
    _set_env(monkeypatch)
    settings = Settings.from_env()

    assert settings.harrier_api_key == "hf_test_token"
    assert settings.harrier_api_url.endswith("microsoft/harrier-oss-v1-0.6b")
    assert settings.harrier_prompt_name == "web_search_query"
    assert settings.harrier_normalize is True

    assert settings.multimodal_api_key == "hf_test_token"
    assert settings.multimodal_api_url == "https://router.huggingface.co/v1/chat/completions"
    assert settings.multimodal_model == "Qwen/Qwen2.5-VL-3B-Instruct"

    assert settings.llm_api_key == "hf_test_token"
    assert settings.llm_api_url == "https://router.huggingface.co/v1/chat/completions"
    assert settings.llm_model == "openai/gpt-oss-20b:fastest"

    # Creative image generation uses the same HF token via InferenceClient.
    assert settings.image_gen_api_style == "hf_hub"
    assert settings.image_gen_api_key == "hf_test_token"
    assert settings.image_gen_model == "black-forest-labs/FLUX.1-schnell"


def test_old_vlm_env_names_are_accepted_as_migration_aliases(monkeypatch):
    values = dict(BASE_ENV)
    values.update({
        "VLM_MODEL": "Qwen/Qwen2.5-VL-3B-Instruct",
        "VLM_API_URL": "https://router.huggingface.co/v1/chat/completions",
    })
    _set_env(monkeypatch, values)
    settings = Settings.from_env()
    assert settings.multimodal_model == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert settings.multimodal_api_key == "hf_test_token"


def test_placeholder_old_vlm_endpoint_is_ignored(monkeypatch):
    values = dict(BASE_ENV)
    values["VLM_API_URL"] = "https://your-vlm-endpoint.endpoints.huggingface.cloud"
    _set_env(monkeypatch, values)
    settings = Settings.from_env()
    assert settings.multimodal_api_url == "https://router.huggingface.co/v1/chat/completions"


def test_hf_reranker_endpoint_reuses_hf_token(monkeypatch):
    values = dict(BASE_ENV)
    values["RERANKER_API_URL"] = "https://reranker.endpoints.huggingface.cloud"
    _set_env(monkeypatch, values)
    settings = Settings.from_env()
    assert settings.reranker_api_key == "hf_test_token"
    assert settings.reranker_api_style == "hf_tei"


def test_provider_specific_harrier_key_overrides_hf_token(monkeypatch):
    values = dict(BASE_ENV)
    values["HARRIER_API_KEY"] = "dedicated_harrier_key"
    _set_env(monkeypatch, values)
    assert Settings.from_env().harrier_api_key == "dedicated_harrier_key"


def test_hf_token_not_forwarded_to_non_hf_multimodal_endpoint(monkeypatch):
    values = dict(BASE_ENV)
    values["QWEN_VL_API_URL"] = "https://vision.example.com/infer"
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
    assert provider._payload(["flood mitigation"], prompt_name=provider.prompt_name) == {
        "inputs": ["flood mitigation"], "normalize": True, "prompt_name": "web_search_query"
    }
    assert provider._payload(["document text"], prompt_name=None) == {
        "inputs": ["document text"], "normalize": True
    }
