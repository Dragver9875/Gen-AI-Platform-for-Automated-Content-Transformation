from __future__ import annotations
import contextlib
import contextvars
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TelemetryEvent:
    provider: str
    operation: str
    latency_ms: float
    success: bool = True
    attempts: int = 1
    input_tokens: int = 0
    output_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class TelemetryCollector:
    def __init__(self):
        self.events: list[TelemetryEvent] = []

    def record(self, event: TelemetryEvent) -> None:
        self.events.append(event)

    def record_usage(self, provider: str, operation: str, *, input_tokens: int = 0, output_tokens: int = 0, metadata: dict[str, Any] | None = None) -> None:
        self.record(TelemetryEvent(provider, operation, 0.0, True, 0, int(input_tokens or 0), int(output_tokens or 0), metadata or {}))

    def summary(self, pricing: dict[str, Any] | None = None) -> dict[str, Any]:
        pricing = pricing or {}
        latencies = sorted(e.latency_ms for e in self.events if e.latency_ms > 0)
        def pct(p: float) -> float:
            if not latencies:
                return 0.0
            idx = min(len(latencies) - 1, max(0, round((len(latencies) - 1) * p)))
            return float(latencies[idx])
        providers: dict[str, dict[str, Any]] = {}
        estimated_cost = 0.0
        for e in self.events:
            item = providers.setdefault(e.provider, {"requests": 0, "errors": 0, "latency_ms": 0.0, "attempts": 0, "input_tokens": 0, "output_tokens": 0})
            if e.attempts:
                item["requests"] += 1
            if not e.success:
                item["errors"] += 1
            item["latency_ms"] += e.latency_ms
            item["attempts"] += e.attempts
            item["input_tokens"] += e.input_tokens
            item["output_tokens"] += e.output_tokens
        for name, item in providers.items():
            cfg = pricing.get(name, {}) if isinstance(pricing, dict) else {}
            estimated_cost += (item["input_tokens"] / 1_000_000) * float(cfg.get("input_per_million", 0.0) or 0.0)
            estimated_cost += (item["output_tokens"] / 1_000_000) * float(cfg.get("output_per_million", 0.0) or 0.0)
            estimated_cost += item["requests"] * float(cfg.get("per_request", 0.0) or 0.0)
        return {
            "provider_request_count": sum(v["requests"] for v in providers.values()),
            "provider_error_count": sum(v["errors"] for v in providers.values()),
            "provider_retry_count": sum(max(0, v["attempts"] - v["requests"]) for v in providers.values()),
            "api_latency_p50_ms": pct(0.50),
            "api_latency_p95_ms": pct(0.95),
            "input_tokens": sum(v["input_tokens"] for v in providers.values()),
            "output_tokens": sum(v["output_tokens"] for v in providers.values()),
            "estimated_cost": estimated_cost,
            "providers": providers,
        }


_current: contextvars.ContextVar[TelemetryCollector | None] = contextvars.ContextVar("eval_telemetry", default=None)


def current_collector() -> TelemetryCollector | None:
    return _current.get()


def record_event(event: TelemetryEvent) -> None:
    collector = current_collector()
    if collector is not None:
        collector.record(event)


def record_usage(provider: str, operation: str, **kwargs: Any) -> None:
    collector = current_collector()
    if collector is not None:
        collector.record_usage(provider, operation, **kwargs)


@contextlib.contextmanager
def telemetry_session():
    collector = TelemetryCollector()
    token = _current.set(collector)
    started = time.perf_counter()
    try:
        yield collector
    finally:
        collector.wall_time_ms = (time.perf_counter() - started) * 1000.0  # type: ignore[attr-defined]
        _current.reset(token)
