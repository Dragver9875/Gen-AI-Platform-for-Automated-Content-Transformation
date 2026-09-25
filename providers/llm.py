from __future__ import annotations

import json
import re
from typing import Any

from providers.http import APIClient, ProviderError


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.IGNORECASE | re.DOTALL)


class HostedLLMProvider:
    """Provider-agnostic hosted text-generation adapter.

    Supported endpoint styles:
      * openai: OpenAI-compatible chat-completions endpoint.
      * hf: Hugging Face / dedicated text-generation endpoint using {"inputs": prompt}.

    `api_url` is always the exact POST endpoint. No model weights are loaded here.
    """

    def __init__(
        self,
        api_url: str,
        api_key: str,
        *,
        api_style: str = "openai",
        model: str | None = None,
        response_mode: str = "json_object",
        timeout_s: float = 120.0,
        retries: int = 2,
    ):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.api_style = api_style.lower()
        self.model = model
        self.response_mode = response_mode.lower()
        self.http = APIClient(timeout_s=timeout_s, retries=retries)

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
        payload = self._payload(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema_name=schema_name,
            json_schema=json_schema,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        response = self.http.request("POST", self.api_url, headers=self.headers, json=payload)
        text = self._extract_text(response.json())
        return self._parse_json(text)

    def _payload(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema_name: str,
        json_schema: dict[str, Any],
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        if self.api_style == "openai":
            payload: dict[str, Any] = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if self.model:
                payload["model"] = self.model
            if self.response_mode == "json_schema":
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": schema_name, "schema": json_schema, "strict": True},
                }
            elif self.response_mode == "json_object":
                payload["response_format"] = {"type": "json_object"}
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
