from __future__ import annotations

import json
import re
from typing import Any

from providers.http import APIClient, ProviderError
from core.telemetry import record_usage


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.IGNORECASE | re.DOTALL)


class HostedLLMProvider:
    """Provider-agnostic hosted open-weight LLM adapter.

    Supported endpoint styles:
      * openai: OpenAI-compatible chat-completions endpoint.
      * hf: Hugging Face / dedicated text-generation endpoint using {"inputs": prompt}.

    `api_url` is always the exact POST endpoint. No model weights are loaded here.

    OpenAI-compatible providers differ in structured-output support. For JSON
    generation the adapter therefore degrades safely:

        json_schema -> json_object -> prompt_only

    The system prompt already contains the JSON Schema, so prompt-only mode still
    preserves the application's schema validation boundary after generation.
    """

    def __init__(
        self,
        api_url: str,
        api_key: str,
        *,
        api_style: str = "openai",
        model: str | None = None,
        response_mode: str = "json_schema",
        timeout_s: float = 120.0,
        retries: int = 2,
    ):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.api_style = api_style.lower()
        self.model = model
        self.response_mode = response_mode.lower()
        self.http = APIClient(timeout_s=timeout_s, retries=retries, provider_name="llm")

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema_name: str,
        json_schema: dict[str, Any],
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        modes = self._response_modes()
        failures: list[str] = []

        for mode in modes:
            payload = self._payload(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema_name=schema_name,
                json_schema=json_schema,
                temperature=temperature,
                max_tokens=max_tokens,
                response_mode=mode,
            )
            try:
                response = self.http.request("POST", self.api_url, headers=self.headers, json=payload)
            except ProviderError as exc:
                failures.append(f"{mode}: {exc}")
                # Unsupported response_format fields commonly surface as 400/422.
                # Continue with the less provider-specific payload. Other failures
                # should not be hidden behind multiple semantic retries.
                if (
                    self.api_style == "openai"
                    and mode != "prompt_only"
                    and exc.status_code in {400, 404, 422}
                ):
                    continue
                raise

            response_data = response.json()
            if isinstance(response_data, dict) and isinstance(response_data.get("usage"), dict):
                usage = response_data["usage"]
                record_usage(
                    "llm",
                    "generation",
                    input_tokens=int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
                    output_tokens=int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
                    metadata={"model": self.model or "", "response_mode": mode},
                )
            text = self._extract_text(response_data)
            return self._parse_json(text)

        raise ProviderError("All structured-output request modes failed: " + " | ".join(failures))

    def _response_modes(self) -> list[str]:
        if self.api_style != "openai":
            return [self.response_mode]

        mode = self.response_mode
        if mode == "json_schema":
            return ["json_schema", "json_object", "prompt_only"]
        if mode == "json_object":
            return ["json_object", "prompt_only"]
        if mode in {"prompt_only", "none", "text"}:
            return ["prompt_only"]
        # Unknown values degrade to prompt-only rather than emitting an invalid
        # provider-specific response_format object.
        return ["prompt_only"]

    def _payload(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema_name: str,
        json_schema: dict[str, Any],
        temperature: float,
        max_tokens: int,
        response_mode: str | None = None,
    ) -> dict[str, Any]:
        mode = (response_mode or self.response_mode).lower()

        if self.api_style == "openai":
            payload: dict[str, Any] = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
            }
            if self.model:
                payload["model"] = self.model
            if mode == "json_schema":
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": schema_name, "schema": json_schema, "strict": True},
                }
            elif mode == "json_object":
                payload["response_format"] = {"type": "json_object"}
            # prompt_only deliberately sends no response_format. The system prompt
            # already instructs the model to return only schema-valid JSON.
            return payload

        if self.api_style == "hf":
            schema_text = json.dumps(json_schema, ensure_ascii=False)
            prompt = (
                f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}\n\n"
                f"Return JSON only. Schema name: {schema_name}. Schema: {schema_text}\n\nASSISTANT:\n"
            )
            return {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                    "return_full_text": False,
                },
            }

        raise ProviderError(f"Unsupported LLM_API_STYLE: {self.api_style}")

    @staticmethod
    def _extract_text(data: Any) -> str:
        # OpenAI-compatible chat completions.
        if isinstance(data, dict) and isinstance(data.get("choices"), list) and data["choices"]:
            choice = data["choices"][0]
            if isinstance(choice, dict):
                message = choice.get("message")
                if isinstance(message, dict) and isinstance(message.get("content"), str):
                    return message["content"]
                if isinstance(choice.get("text"), str):
                    return choice["text"]

        # Common HF text-generation response.
        if isinstance(data, list) and data and isinstance(data[0], dict):
            if isinstance(data[0].get("generated_text"), str):
                return data[0]["generated_text"]

        for key in ("generated_text", "output_text", "text", "content"):
            if isinstance(data, dict) and isinstance(data.get(key), str):
                return data[key]

        if isinstance(data, str):
            return data
        raise ProviderError(f"Unsupported LLM response shape: {type(data)!r}")

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        candidate = text.strip()
        match = _FENCE_RE.match(candidate)
        if match:
            candidate = match.group(1).strip()
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError as exc:
            # Conservative recovery for providers that prepend a short sentence.
            first = candidate.find("{")
            last = candidate.rfind("}")
            if first < 0 or last <= first:
                raise ProviderError(f"LLM did not return valid JSON: {exc}") from exc
            try:
                data = json.loads(candidate[first : last + 1])
            except json.JSONDecodeError as second:
                raise ProviderError(f"LLM did not return valid JSON: {second}") from second
        if not isinstance(data, dict):
            raise ProviderError("Structured LLM response must be a JSON object")
        return data
