from __future__ import annotations

import time
from typing import Any

import httpx


class ProviderError(RuntimeError):
    pass


class APIClient:
    def __init__(self, *, timeout_s: float = 90.0, retries: int = 2):
        self.timeout_s = timeout_s
        self.retries = retries

    def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                with httpx.Client(timeout=self.timeout_s, follow_redirects=True) as client:
                    response = client.request(method, url, **kwargs)
                response.raise_for_status()
                return response
            except (httpx.HTTPError, OSError) as exc:
                last_error = exc
                if attempt >= self.retries:
                    break
                time.sleep(min(2**attempt, 4))
        raise ProviderError(f"Request failed after {self.retries + 1} attempt(s): {last_error}")
