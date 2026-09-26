from __future__ import annotations

import base64
import io

from PIL import Image

from providers.http import ProviderError
from providers.reranker import HostedReranker
from providers.vlm import VLMProvider


class _FakeResponse:
    def __init__(self, data):
        self._data = data
    def json(self):
        return self._data


def test_qwen25_vl_openai_multimodal_contract():
    provider = VLMProvider(
        "https://router.huggingface.co/v1/chat/completions",
        "hf_test",
        model="Qwen/Qwen2.5-VL-3B-Instruct",
    )
    seen = {}

    def fake_request(method, url, **kwargs):
        seen.update({"method": method, "url": url, **kwargs})
        return _FakeResponse({"choices": [{"message": {"content": "Visible document text."}}]})

    provider.http.request = fake_request  # type: ignore[method-assign]
    result = provider.describe_bytes(b"fake-png", media_type="image/png", prompt="Transcribe faithfully")

    assert result == "Visible document text."
    assert seen["url"] == "https://router.huggingface.co/v1/chat/completions"
    assert seen["json"]["model"] == "Qwen/Qwen2.5-VL-3B-Instruct"
    content = seen["json"]["messages"][0]["content"]
    assert content[0]["text"] == "Transcribe faithfully"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert seen["headers"]["Authorization"] == "Bearer hf_test"


def test_hf_tei_reranker_contract_and_sorting():
    provider = HostedReranker(
        "https://reranker.endpoints.huggingface.cloud",
        "hf_test",
        api_style="hf_tei",
    )
    seen = {}

    def fake_request(method, url, **kwargs):
        seen.update({"method": method, "url": url, **kwargs})
        return _FakeResponse([
            {"index": 1, "score": 0.95},
            {"index": 0, "score": 0.25},
        ])

    provider.http.request = fake_request  # type: ignore[method-assign]
    docs = [{"text": "alpha", "id": "a"}, {"text": "beta", "id": "b"}]
    result = provider.rerank("query", docs, top_k=2)

    assert seen["url"].endswith("/rerank")
    assert seen["json"] == {"query": "query", "texts": ["alpha", "beta"], "raw_scores": False}
    assert seen["headers"]["Authorization"] == "Bearer hf_test"
    assert [d["id"] for d in result] == ["b", "a"]



def test_flux_hf_hub_path_needs_no_endpoint(monkeypatch):
    import sys
    import types
    from PIL import Image
    from providers.image_generation import HostedImageGenerationProvider

    seen = {}

    class FakeInferenceClient:
        def __init__(self, *, api_key, provider):
            seen["api_key"] = api_key
            seen["provider"] = provider

        def text_to_image(self, prompt, *, model, width, height, negative_prompt=None):
            seen.update({"prompt": prompt, "model": model, "width": width, "height": height})
            return Image.new("RGB", (16, 16), "white")

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(InferenceClient=FakeInferenceClient))
    provider = HostedImageGenerationProvider(
        None,
        "hf_test",
        model="black-forest-labs/FLUX.1-schnell",
        api_style="hf_hub",
        provider="auto",
    )
    data, media_type = provider.generate("clean abstract illustration", width=512, height=512)
    assert data.startswith(b"\x89PNG")
    assert media_type == "image/png"
    assert seen["api_key"] == "hf_test"
    assert seen["provider"] == "auto"
    assert seen["model"] == "black-forest-labs/FLUX.1-schnell"



def _large_test_image(width=2200, height=1800):
    image = Image.effect_noise((width, height), 80).convert("RGB")
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def test_qwen25_vl_bounds_visual_payload_before_request():
    provider = VLMProvider(
        "https://router.huggingface.co/v1/chat/completions",
        "hf_test",
        max_image_side=900,
        max_image_bytes=220_000,
        jpeg_quality=75,
    )
    seen = {}

    def fake_request(method, url, **kwargs):
        seen.update(kwargs)
        return _FakeResponse({"choices": [{"message": {"content": "ok"}}]})

    provider.http.request = fake_request  # type: ignore[method-assign]
    result = provider.describe_bytes(_large_test_image(), media_type="image/png", prompt="Read it")

    assert result == "ok"
    image_url = seen["json"]["messages"][0]["content"][1]["image_url"]["url"]
    assert image_url.startswith("data:image/jpeg;base64,")
    raw = base64.b64decode(image_url.split(",", 1)[1])
    assert len(raw) <= 220_000
    assert provider.last_payload_bytes == len(raw)
    assert provider.last_payload_reduced is True


def test_qwen25_vl_retries_413_with_smaller_payload():
    provider = VLMProvider(
        "https://router.huggingface.co/v1/chat/completions",
        "hf_test",
        max_image_side=1400,
        max_image_bytes=700_000,
        jpeg_quality=82,
        retry_image_side=700,
        retry_image_bytes=180_000,
        retry_jpeg_quality=60,
    )
    payload_sizes = []

    def fake_request(method, url, **kwargs):
        image_url = kwargs["json"]["messages"][0]["content"][1]["image_url"]["url"]
        payload_sizes.append(len(base64.b64decode(image_url.split(",", 1)[1])))
        if len(payload_sizes) == 1:
            raise ProviderError("HTTP 413 from provider", status_code=413)
        return _FakeResponse({"choices": [{"message": {"content": "recovered"}}]})

    provider.http.request = fake_request  # type: ignore[method-assign]
    result = provider.describe_bytes(_large_test_image(), media_type="image/png", prompt="Read it")

    assert result == "recovered"
    assert len(payload_sizes) == 2
    assert payload_sizes[1] < payload_sizes[0]
    assert payload_sizes[1] <= 180_000
    assert provider.last_413_retry is True
