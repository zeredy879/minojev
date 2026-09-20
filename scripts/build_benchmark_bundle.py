#!/usr/bin/env python3
"""Assemble web/data/benchmark.json from committed comparison reports."""

from __future__ import annotations

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "results" / "logits-zero-shot.json"


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def build_workload(name: str, path: Path) -> dict | None:
    report = load(path)
    if not report:
        return None
    engine = report["engine"]
    generative = report["generative"]
    return {
        "name": name,
        "requests": report["requests"],
        "decisions": report["decisions"],
        "engine": {
            "accuracy": engine.get("accuracy"),
            "teacher_topset_accuracy": engine.get("teacher_topset_accuracy"),
            "ece": engine.get("expected_calibration_error"),
            "gold_nll": engine.get("gold_nll"),
            "output_tokens_per_decision": engine.get("output_tokens_per_decision"),
            "input_tokens_per_decision": engine.get("input_tokens_per_decision"),
            "ttft_p50_ms": 0.0,
            "latency_p50_ms": engine.get("latency_per_request_p50_ms") or engine.get("latency_p50_ms"),
            "latency_p95_ms": engine.get("latency_per_request_p95_ms") or engine.get("latency_p95_ms"),
            "decisions_per_second": engine.get("decisions_per_second"),
            "parse_failure_rate": engine.get("parse_failure_rate"),
            "selective_accuracy": engine.get("selective_accuracy"),
        },
        "generative": {
            "accuracy": generative.get("accuracy"),
            "accuracy_when_parsed": generative.get("accuracy_when_parsed"),
            "output_tokens_per_decision": generative.get("output_tokens_per_decision"),
            "input_tokens_per_decision": generative.get("input_tokens_per_decision"),
            "ttft_p50_ms": generative.get("ttft_p50_ms"),
            "latency_p50_ms": generative.get("latency_per_request_p50_ms") or generative.get("latency_p50_ms"),
            "latency_p95_ms": generative.get("latency_per_request_p95_ms") or generative.get("latency_p95_ms"),
            "decisions_per_second": generative.get("decisions_per_second"),
            "parse_failure_rate": generative.get("parse_failure_rate"),
        },
        "per_source": report.get("per_source", {}),
    }


def main() -> int:
    bundle = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": "Qwen3-1.7B + trained decision head (bf16)",
        "reference": load(REFERENCE) or {},
        "workloads": [],
    }
    workloads = [
        ("general (in-domain)", ROOT / "results" / "compare-general.json"),
        ("domains (3 questions/request, OOD)", ROOT / "results" / "compare-domains.json"),
    ]
    for name, path in workloads:
        entry = build_workload(name, path)
        if entry:
            bundle["workloads"].append(entry)
    output = ROOT / "web" / "data" / "benchmark.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(output), "workloads": len(bundle["workloads"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
