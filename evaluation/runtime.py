from __future__ import annotations
from typing import Any


def runtime_metrics(telemetry: dict[str, Any], wall_time_ms: float) -> dict[str, Any]:
    return {
        "end_to_end_latency_ms": float(wall_time_ms),
        "provider_request_count": int(telemetry.get("provider_request_count") or 0),
        "provider_error_count": int(telemetry.get("provider_error_count") or 0),
        "provider_retry_count": int(telemetry.get("provider_retry_count") or 0),
        "api_latency_p50_ms": float(telemetry.get("api_latency_p50_ms") or 0.0),
        "api_latency_p95_ms": float(telemetry.get("api_latency_p95_ms") or 0.0),
        "input_tokens": int(telemetry.get("input_tokens") or 0),
        "output_tokens": int(telemetry.get("output_tokens") or 0),
        "estimated_cost": float(telemetry.get("estimated_cost") or 0.0),
    }
