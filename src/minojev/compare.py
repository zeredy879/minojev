"""Side-by-side benchmark: zero-decoding decisions versus token generation.

Both engines answer the same frozen requests. The report compares decision
quality (accuracy, calibration) with token economy and wall-clock behaviour
(output tokens per decision, time to first token, p50/p95 latency).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from .generative import GenerativeOptions, run_generative_baseline, summarize_generative
from .metrics import aggregate
from .model import DecisionModel, ScoreOptions
from .types import Request


@dataclass
class CompareOptions:
    limit: int = 0
    mode: str = "fresh"
    max_new_tokens: int = 12
    device: str = "auto"
    chat_template: bool = False


def _latency_summary(seconds: list[float]) -> dict:
    if not seconds:
        return {"latency_p50_ms": None, "latency_p95_ms": None, "latency_mean_ms": None}
    ordered = sorted(seconds)
    return {
        "latency_p50_ms": round(1000 * ordered[len(ordered) // 2], 3),
        "latency_p95_ms": round(1000 * ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 3),
        "latency_mean_ms": round(1000 * sum(ordered) / len(ordered), 3),
    }


def select_balanced(requests: list[Request], limit: int) -> list[Request]:
    """Round-robin across sources so a small limit still covers every dataset."""
    if not limit or limit >= len(requests):
        return requests
    buckets: dict[str, list[Request]] = {}
    order: list[str] = []
    for request in requests:
        family = request.questions[0].family or request.id.rsplit("-", 1)[0]
        if family not in buckets:
            buckets[family] = []
            order.append(family)
        buckets[family].append(request)
    selected: list[Request] = []
    index = 0
    while len(selected) < limit:
        added = False
        for family in order:
            bucket = buckets[family]
            if index < len(bucket):
                selected.append(bucket[index])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        index += 1
    return selected


def score_workload(model: DecisionModel, requests: list[Request], options: CompareOptions) -> tuple[list[dict], dict]:
    model.eval_mode(options.device)
    records: list[dict] = []
    latencies: list[float] = []
    if requests:  # warm up the device on a copy so every measured request is kept
        model.score([requests[0]], ScoreOptions(mode=options.mode, batch_requests=1, device=options.device))
    for request in requests:
        started = time.perf_counter()
        scored = model.score([request], ScoreOptions(mode=options.mode, batch_requests=1, device=options.device))
        elapsed = time.perf_counter() - started
        latencies.append(elapsed)
        records.extend(scored)
    summary = aggregate(records)
    summary.update(_latency_summary(latencies))
    input_tokens = [record.get("input_tokens") for record in records if record.get("input_tokens") is not None]
    summary.update(
        {
            "output_tokens_per_decision": 0.0,
            "decode_steps_per_decision": 0.0,
            "parse_failure_rate": 0.0,
            "decisions_per_second": round(len(records) / max(sum(latencies), 1e-9), 2),
            "input_tokens_per_decision": round(sum(input_tokens) / len(input_tokens), 1) if input_tokens else None,
            "latency_per_request_p50_ms": summary["latency_p50_ms"],
            "latency_per_request_p95_ms": summary["latency_p95_ms"],
        }
    )
    return records, summary


def compare_workload(
    model: DecisionModel,
    requests: list[Request],
    generative_model,
    tokenizer,
    options: CompareOptions | None = None,
) -> dict:
    options = options or CompareOptions()
    selected = select_balanced(requests, options.limit)
    engine_records, engine_summary = score_workload(model, selected, options)
    generative_records = run_generative_baseline(
        generative_model,
        tokenizer,
        selected,
        GenerativeOptions(
            max_new_tokens=options.max_new_tokens,
            device=options.device,
            chat_template=options.chat_template,
        ),
    )
    generative_summary = summarize_generative(generative_records)
    by_source = {}
    for record in engine_records:
        family = record.get("family") or "unknown"
        bucket = by_source.setdefault(family, {"engine": [], "generative": []})
        if "correct" in record:
            bucket["engine"].append(bool(record["correct"]))
    for record in generative_records:
        family = record.get("family") or "unknown"
        bucket = by_source.setdefault(family, {"engine": [], "generative": []})
        if "correct" in record:
            bucket["generative"].append(bool(record["correct"]))
    per_source = {}
    for family, bucket in by_source.items():
        entry = {}
        if bucket["engine"]:
            entry["engine_accuracy"] = round(sum(bucket["engine"]) / len(bucket["engine"]), 6)
        if bucket["generative"]:
            entry["generative_accuracy"] = round(sum(bucket["generative"]) / len(bucket["generative"]), 6)
        per_source[family] = entry
    return {
        "requests": len(selected),
        "decisions": len(engine_records),
        "engine": engine_summary,
        "generative": generative_summary,
        "per_source": per_source,
    }


def render_markdown(report: dict, engine_label: str = "minojev", generative_label: str = "generative baseline") -> str:
    engine = report["engine"]
    generative = report["generative"]
    rows = [
        ("accuracy", _percent(engine.get("accuracy")), _percent(generative.get("accuracy"))),
        ("teacher top-set", _percent(engine.get("teacher_topset_accuracy")), "—"),
        ("ECE", _number(engine.get("expected_calibration_error")), "—"),
        ("gold NLL", _number(engine.get("gold_nll")), "—"),
        ("input tokens / decision", _number(engine.get("input_tokens_per_decision")), _number(generative.get("input_tokens_per_decision"))),
        ("output tokens / decision", _number(engine.get("output_tokens_per_decision")), _number(generative.get("output_tokens_per_decision"))),
        ("decode steps / decision", _number(engine.get("decode_steps_per_decision")), _number(generative.get("output_tokens_per_decision"))),
        ("time to first token (p50)", "0 ms", _ms(generative.get("ttft_p50_ms"))),
        ("latency per request p50", _ms(engine.get("latency_per_request_p50_ms")), _ms(generative.get("latency_per_request_p50_ms"))),
        ("latency per request p95", _ms(engine.get("latency_per_request_p95_ms")), _ms(generative.get("latency_per_request_p95_ms"))),
        ("decisions / second", _number(engine.get("decisions_per_second")), _number(generative.get("decisions_per_second"))),
        ("format / parse failures", "0", _percent(generative.get("parse_failure_rate"))),
    ]
    lines = [
        f"# Decision engine vs generative baseline",
        "",
        f"Workload: {report['requests']} requests, {report['decisions']} decisions.",
        "",
        f"| metric | {engine_label} | {generative_label} |",
        "|---|---:|---:|",
    ]
    lines.extend(f"| {name} | {left} | {right} |" for name, left, right in rows)
    lines.append("")
    lines.append("| source | engine accuracy | generative accuracy |")
    lines.append("|---|---:|---:|")
    for family, entry in sorted(report.get("per_source", {}).items()):
        lines.append(
            f"| {family} | {_percent(entry.get('engine_accuracy'))} | {_percent(entry.get('generative_accuracy'))} |"
        )
    lines.append("")
    return "\n".join(lines)


def _percent(value) -> str:
    return "—" if value is None else f"{100 * float(value):.1f}%"


def _number(value) -> str:
    return "—" if value is None else f"{float(value):.3f}"


def _ms(value) -> str:
    return "—" if value is None else f"{float(value):.1f} ms"


def write_report(report: dict, json_path: str | Path, markdown_path: str | Path | None = None, labels: tuple[str, str] = ("engine", "generative")) -> None:
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if markdown_path:
        markdown_path = Path(markdown_path)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_markdown(report, *labels) + "\n")
