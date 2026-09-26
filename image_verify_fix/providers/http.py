from __future__ import annotations

import time
from typing import Any

import httpx

from core.telemetry import TelemetryEvent, record_event


class ProviderError(RuntimeError):
    """Normalized provider failure with optional HTTP diagnostics."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        response_text: str | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response_text = response_text


class APIClient:
    def __init__(self, *, timeout_s: float = 90.0, retries: int = 2, provider_name: str = "http"):
        self.timeout_s = timeout_s
        self.retries = retries
        self.provider_name = provider_name

    @staticmethod
    def _retryable_status(status_code: int) -> bool:
        # Retrying malformed/unsupported 4xx payloads only adds latency. Retry
        # transient throttling/timeouts and server-side failures instead.
        return status_code in {408, 409, 425, 429} or status_code >= 500

    @staticmethod
    def _body_preview(response: httpx.Response | None, *, max_chars: int = 2000) -> str | None:
        if response is None:
            return None
        try:
            text = response.text.strip()
        except Exception:
            return None
        if not text:
            return None
        return text[:max_chars]

    def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        last_error: Exception | None = None
        last_status: int | None = None
        last_body: str | None = None
        started = time.perf_counter()
        attempts = 0

        for attempt in range(self.retries + 1):
            attempts = attempt + 1
            try:
                with httpx.Client(timeout=self.timeout_s, follow_redirects=True) as client:
                    response = client.request(method, url, **kwargs)
                response.raise_for_status()
                record_event(TelemetryEvent(
                    provider=self.provider_name,
                    operation=method.upper(),
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                    success=True,
                    attempts=attempts,
                    metadata={"status_code": response.status_code},
                ))
                return response
            except httpx.HTTPStatusError as exc:
                last_error = exc
                last_status = exc.response.status_code if exc.response is not None else None
                last_body = self._body_preview(exc.response)

                # Do not retry permanent client-side request errors. This also lets
                # higher-level adapters degrade unsupported request features quickly.
                if last_status is not None and not self._retryable_status(last_status):
                    record_event(TelemetryEvent(
                        provider=self.provider_name,
                        operation=method.upper(),
                        latency_ms=(time.perf_counter() - started) * 1000.0,
                        success=False,
                        attempts=attempts,
                        metadata={
                            "status_code": last_status,
                            "error": str(last_error),
                            "response": last_body or "",
                        },
                    ))
                    detail = f"HTTP {last_status} from provider"
                    if last_body:
                        detail += f": {last_body}"
                    raise ProviderError(
                        detail,
                        status_code=last_status,
                        response_text=last_body,
                    ) from exc

                if attempt >= self.retries:
                    break
                time.sleep(min(2**attempt, 4))
            except (httpx.HTTPError, OSError) as exc:
                last_error = exc
                if attempt >= self.retries:
                    break
                time.sleep(min(2**attempt, 4))

        record_event(TelemetryEvent(
            provider=self.provider_name,
            operation=method.upper(),
            latency_ms=(time.perf_counter() - started) * 1000.0,
            success=False,
            attempts=attempts,
            metadata={
                "status_code": last_status or 0,
                "error": str(last_error),
                "response": last_body or "",
            },
        ))
        detail = f"Request failed after {attempts} attempt(s): {last_error}"
        if last_body:
            detail += f"; response={last_body}"
        raise ProviderError(detail, status_code=last_status, response_text=last_body)
