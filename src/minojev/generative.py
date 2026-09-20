"""Generative baseline for decision workloads.

The same requests are answered by decoding tokens instead of reading hidden
states, so token economy and latency can be compared directly against the
zero-decoding engine. Decoding is manual and greedy: one forward per output
token, which also yields time-to-first-token.
"""

from __future__ import annotations

import re
import string
import time
from dataclasses import dataclass
from typing import Any

import torch

from .types import BOOLEAN, Request, serialize_state

LETTERS = string.ascii_uppercase


def option_letters(count: int) -> str:
    if count > len(LETTERS):
        raise ValueError(f"At most {len(LETTERS)} lettered options are supported")
    return LETTERS[:count]


def generative_prompt(request: Request, question: Any) -> str:
    lines = [
        "State:",
        serialize_state(request.state),
        "Question:",
        question.instructions,
        "Options:",
    ]
    for index, candidate in enumerate(question.candidates):
        lines.append(f"{LETTERS[index]}. {candidate.description}")
    lines.append('Answer with only the option letter, for example "A".')
    return "\n".join(lines)


def parse_answer(text: str, question: Any) -> tuple[str | None, str | None]:
    """Extract a candidate id from generated text.

    Returns ``(candidate_id, method)`` where method is ``letter``,
    ``text``, or ``None`` when nothing could be parsed.
    """
    stripped = text.strip()
    if not stripped:
        return None, None
    if question.kind == BOOLEAN:
        lowered = stripped.lower()
        for value, candidate_id in ((True, "true"), (False, "false")):
            words = ("true", "yes") if value else ("false", "no")
            if any(word in lowered for word in words):
                return candidate_id, "boolean"
    letters = LETTERS[: len(question.candidates)]
    # Only standalone letters count: "The answer is C." must not match the A in "answer".
    match = re.search(rf"\b([{letters}])\b", stripped.upper())
    if match:
        return question.candidates[LETTERS.index(match.group(1))].candidate_id, "letter"
    lowered = stripped.lower()
    matches = [
        candidate.candidate_id
        for candidate in question.candidates
        if candidate.candidate_id.lower() in lowered
    ]
    if matches:
        longest = max(matches, key=len)
        return longest, "text"
    return None, None


@dataclass
class GenerativeOptions:
    max_new_tokens: int = 12
    device: str = "auto"
    chat_template: bool = False


def _resolve_device(device: str, model=None) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if model is not None:
        try:
            return next(model.parameters()).device
        except StopIteration:  # pragma: no cover - defensive
            pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def generate_one(model, tokenizer, prompt: str, options: GenerativeOptions) -> dict:
    """Greedy token-by-token decode; returns text and timing."""
    device = _resolve_device(options.device, model)
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None:
        eos_id = getattr(tokenizer, "eos_id", None)
    started = time.perf_counter()
    first_token_seconds = None
    generated: list[int] = []
    past = None
    current = input_ids
    for _ in range(max(1, options.max_new_tokens)):
        output = model(input_ids=current, past_key_values=past, use_cache=True)
        past = output.past_key_values
        next_id = int(output.logits[0, -1].argmax().item())
        if first_token_seconds is None:
            first_token_seconds = time.perf_counter() - started
        generated.append(next_id)
        if eos_id is not None and next_id == eos_id:
            break
        current = torch.tensor([[next_id]], dtype=torch.long, device=device)
    total_seconds = time.perf_counter() - started
    try:
        text = tokenizer.decode(generated, skip_special_tokens=True)
    except TypeError:  # ByteTokenizer has no skip_special_tokens argument
        text = tokenizer.decode(generated)
    return {
        "text": text,
        "input_tokens": int(input_ids.shape[1]),
        "output_tokens": len(generated),
        "first_token_seconds": first_token_seconds if first_token_seconds is not None else total_seconds,
        "total_seconds": total_seconds,
    }


def run_generative_baseline(model, tokenizer, requests: list[Request], options: GenerativeOptions | None = None) -> list[dict]:
    options = options or GenerativeOptions()
    model.eval()
    records: list[dict] = []
    for request in requests:
        for question in request.questions:
            prompt = generative_prompt(request, question)
            if options.chat_template and hasattr(tokenizer, "apply_chat_template"):
                messages = [{"role": "user", "content": prompt}]
                try:
                    # Qwen3-family templates support disabling chain-of-thought tokens.
                    prompt = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                    )
                except TypeError:
                    prompt = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
            result = generate_one(model, tokenizer, prompt, options)
            parsed, method = parse_answer(result["text"], question)
            record = {
                "id": request.id,
                "qid": question.qid,
                "type": question.kind,
                "family": question.family or request.state.get("source", "") if isinstance(request.state, dict) else question.family,
                "candidate_ids": question.candidate_ids,
                "parsed": parsed,
                "parse_method": method,
                "parse_ok": parsed is not None,
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "first_token_seconds": round(result["first_token_seconds"], 6),
                "total_seconds": round(result["total_seconds"], 6),
                "generated_text": result["text"][:200],
            }
            if question.qid in request.gold:
                gold = request.gold[question.qid]
                record["gold"] = gold
                if question.kind == BOOLEAN:
                    record["correct"] = parsed in {"true", "false"} and (parsed == "true") == (gold is True or gold in (1, "true", "True"))
                elif parsed is None:
                    record["correct"] = False
                else:
                    record["correct"] = parsed == gold
            records.append(record)
    return records


def summarize_generative(records: list[dict]) -> dict:
    total = len(records)
    labelled = [record for record in records if "correct" in record]
    latencies = sorted(record["total_seconds"] for record in records)
    ttfts = sorted(record["first_token_seconds"] for record in records)
    summary = {
        "decisions": total,
        "parsed": sum(1 for record in records if record["parse_ok"]),
        "parse_failure_rate": round(1 - sum(1 for record in records if record["parse_ok"]) / max(total, 1), 6),
        "output_tokens_per_decision": round(sum(record["output_tokens"] for record in records) / max(total, 1), 3),
        "input_tokens_per_decision": round(sum(record["input_tokens"] for record in records) / max(total, 1), 1),
        "decisions_per_second": round(total / max(sum(record["total_seconds"] for record in records), 1e-9), 2),
        "latency_p50_ms": round(1000 * latencies[len(latencies) // 2], 3) if latencies else None,
        "latency_p95_ms": round(1000 * latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))], 3) if latencies else None,
        "ttft_p50_ms": round(1000 * ttfts[len(ttfts) // 2], 3) if ttfts else None,
    }
    by_request: dict[str, float] = {}
    for record in records:
        by_request[record["id"]] = by_request.get(record["id"], 0.0) + record["total_seconds"]
    request_latencies = sorted(by_request.values())
    if request_latencies:
        summary["latency_per_request_p50_ms"] = round(1000 * request_latencies[len(request_latencies) // 2], 3)
        summary["latency_per_request_p95_ms"] = round(
            1000 * request_latencies[min(len(request_latencies) - 1, int(0.95 * len(request_latencies)))], 3
        )
    if labelled:
        summary["accuracy"] = round(sum(1 for record in labelled if record["correct"]) / len(labelled), 6)
        summary["accuracy_when_parsed"] = round(
            sum(1 for record in labelled if record["correct"]) / max(sum(1 for record in labelled if record["parse_ok"]), 1),
            6,
        )
    return summary
