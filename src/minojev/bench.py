"""Latency and throughput benchmark for decision serving.

Reports per-request latency percentiles (the end-to-end cost of one decision
batch) and batched throughput for both serving modes. A decision is one
question; one scored request may contain several questions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .model import DecisionModel, ScoreOptions
from .types import Request


@dataclass
class BenchOptions:
    modes: tuple[str, ...] = ("fresh", "reuse")
    repeats: int = 3
    max_latency_requests: int = 64
    batch_requests: int = 16
    device: str = "auto"


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def benchmark_model(model: DecisionModel, requests: list[Request], options: BenchOptions | None = None) -> dict:
    options = options or BenchOptions()
    model.eval_mode(options.device)
    serial = requests[: options.max_latency_requests]
    results = {}
    for mode in options.modes:
        model.score(requests[:1], ScoreOptions(mode=mode, batch_requests=1, device=options.device))
        latencies = []
        question_counts = []
        for _ in range(options.repeats):
            for request in serial:
                began = time.perf_counter()
                records = model.score([request], ScoreOptions(mode=mode, batch_requests=1, device=options.device))
                latencies.append(time.perf_counter() - began)
                question_counts.append(len(records))
        began = time.perf_counter()
        batched = model.score(
            requests, ScoreOptions(mode=mode, batch_requests=options.batch_requests, device=options.device)
        )
        batch_seconds = time.perf_counter() - began
        results[mode] = {
            "requests_timed": len(latencies),
            "questions_timed": sum(question_counts),
            "latency_p50_ms": round(1000 * _percentile(latencies, 0.50), 3),
            "latency_p95_ms": round(1000 * _percentile(latencies, 0.95), 3),
            "latency_mean_ms": round(1000 * sum(latencies) / len(latencies), 3),
            "batch_requests": len(requests),
            "batch_questions": len(batched),
            "batch_seconds": round(batch_seconds, 4),
            "decisions_per_second": round(sum(1 for _ in batched) / batch_seconds, 2),
            "decode_steps": sum(record.get("decode_steps", 0) for record in batched),
        }
    common = {
        "parameters": sum(parameter.numel() for parameter in model.trainable_parameters()),
        "device": str(next(model.head.parameters()).device),
    }
    return {**common, "modes": results}


def compare_modes(result: dict) -> dict:
    modes = result.get("modes", {})
    if "fresh" in modes and "reuse" in modes:
        fresh, reuse = modes["fresh"], modes["reuse"]
        return {
            "throughput_ratio": round(reuse["decisions_per_second"] / max(fresh["decisions_per_second"], 1e-9), 3),
            "latency_p50_ratio": round(reuse["latency_p50_ms"] / max(fresh["latency_p50_ms"], 1e-9), 3),
        }
    return {}
