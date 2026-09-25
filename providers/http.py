from __future__ import annotations

import time
from typing import Any

import httpx

from core.telemetry import TelemetryEvent, record_event


class ProviderError(RuntimeError):
    pass


class APIClient:
    def __init__(self, *, timeout_s: float = 90.0, retries: int = 2, provider_name: str = "http"):
        self.timeout_s = timeout_s
        self.retries = retries
        self.provider_name = provider_name

    def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        last_error: Exception | None = None
        started = time.perf_counter()
        attempts = 0
        for attempt in range(self.retries + 1):
            attempts = attempt + 1
            try:
                with httpx.Client(timeout=self.timeout_s, follow_redirects=True) as client:
                    response = client.request(method, url, **kwargs)
                response.raise_for_status()
                record_event(TelemetryEvent(
                    provider=self.provider_name, operation=method.upper(),
                    latency_ms=(time.perf_counter()-started)*1000.0, success=True, attempts=attempts,
                    metadata={"status_code": response.status_code},
                ))
                return response
            except (httpx.HTTPError, OSError) as exc:
                last_error = exc
                if attempt >= self.retries:
                    break
                time.sleep(min(2**attempt, 4))
        record_event(TelemetryEvent(
            provider=self.provider_name, operation=method.upper(),
            latency_ms=(time.perf_counter()-started)*1000.0, success=False, attempts=attempts,
            metadata={"error": str(last_error)},
        ))
        raise ProviderError(f"Request failed after {self.retries + 1} attempt(s): {last_error}")
