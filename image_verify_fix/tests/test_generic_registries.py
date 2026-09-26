from __future__ import annotations

from pathlib import Path

from ingestion.registry import IngestionProcessorRegistry
from providers.registry import ProviderCapability, ProviderRegistry
from retrieval.config import RetrievalConfig
from verification.profiles import VerificationProfileRegistry


def test_provider_registry_resolves_capabilities():
    registry = ProviderRegistry()
    obj = object()
    registry.register(ProviderCapability.EMBEDDING, obj, name="harrier")
    assert registry.resolve(ProviderCapability.EMBEDDING) is obj


def test_ingestion_registry_can_add_new_file_family():
    registry = IngestionProcessorRegistry()
    handler = lambda path, source_id, media_type: None
    registry.register("json", {".json"}, handler)
    assert registry.resolve(Path("sample.json")).name == "json"


def test_retrieval_config_is_not_hardcoded_to_hybrid():
    cfg = RetrievalConfig(strategy="dense", vector_k=22, final_k=7, use_reranker=False)
    assert cfg.strategy == "dense"
    assert cfg.vector_k == 22
    assert cfg.final_k == 7


def test_verification_profiles_are_configurable():
    registry = VerificationProfileRegistry.from_json('{"regulated":{"min_faithfulness_score":1.0,"allow_partial":false,"require_evidence":true},"creative":{"min_faithfulness_score":0.7,"allow_partial":true,"require_evidence":true}}', default_profile="regulated")
    assert registry.resolve(None).name == "regulated"
    assert registry.resolve("creative").allow_partial is True
