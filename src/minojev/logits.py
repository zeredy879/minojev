"""Native-logits decision readout for pretrained language models.

This is the zero-training route: build one prompt per question, run a single
forward pass, and softmax the next-token logits over the declared answer-slot
tokens. No token is sampled and no answer sentence is parsed.

The route is intended for pretrained causal LMs. It also works mechanically
with the built-in tiny backbone, which is useful for contract tests but is not
expected to make meaningful zero-shot judgments.
"""

from __future__ import annotations

import time

import torch

from .backbone import resolve_device
from .types import BOOLEAN, CHOICE, SCORE, Question, Request, serialize_state

LETTERS = "ABCDEFGHIJKLMNOP"
MAX_SLOTS = len(LETTERS)
PROMPT_VERSION = "minojev-logits-v1"


def letter_slot_ids(tokenizer) -> list[int]:
    slots = []
    for letter in LETTERS:
        encoded = tokenizer.encode(letter)
        if len(encoded) != 1 or tokenizer.decode(encoded) != letter:
            raise ValueError(f"Answer slot {letter!r} is not one exact round-trip token")
        slots.append(encoded[0])
    if len(set(slots)) != len(slots):
        raise ValueError("Answer-slot tokens collide")
    return slots


def build_prompt(request: Request, question: Question) -> str:
    lines = [
        f"State:\n{serialize_state(request.state)}",
        f"Question type: {question.kind}",
        f"Question:\n{question.instructions}",
    ]
    if question.kind == BOOLEAN:
        for key in ("false", "true"):
            if key in question.criteria:
                lines.append(f"{key.capitalize()} criterion: {question.criteria[key]}")
    lines.append("Options:")
    for index, candidate in enumerate(question.candidates):
        lines.append(f"{LETTERS[index]}: {candidate.description}")
    lines.append("Answer with only the option letter.")
    return "\n".join(lines) + "\nAnswer:"


@torch.inference_mode()
def score_logits(backbone, tokenizer, requests: list[Request], device: str = "auto", max_tokens: int = 4096) -> list[dict]:
    if isinstance(backbone, torch.nn.Module):
        backbone.eval()
        backbone.to(resolve_device(device))
    else:
        backbone.model.eval()
    if isinstance(backbone, torch.nn.Module):
        target_device = next(backbone.parameters()).device
    else:
        target_device = next(backbone.model.parameters()).device
    slots = letter_slot_ids(tokenizer)
    records = []
    for request in requests:
        for question in request.questions:
            if len(question.candidates) > MAX_SLOTS:
                raise ValueError(f"Logits readout supports up to {MAX_SLOTS} candidates; use the head engine for larger choices")
            prompt = build_prompt(request, question)
            ids = tokenizer.encode(prompt)
            if len(ids) > max_tokens:
                raise ValueError(f"{request.id}:{question.qid}: {len(ids)} tokens exceed limit {max_tokens}")
            input_ids = torch.tensor([ids], dtype=torch.long, device=target_device)
            started = time.perf_counter()
            hidden, _ = backbone(input_ids=input_ids)
            logits = backbone.logits(hidden)[0, -1].float()
            selected = logits[slots[: len(question.candidates)]].tolist()
            probabilities = torch.softmax(torch.tensor(selected, dtype=torch.float32), dim=-1).tolist()
            elapsed = time.perf_counter() - started
            record = {
                "id": request.id,
                "qid": question.qid,
                "type": question.kind,
                "state": request.state,
                "instructions": question.instructions,
                "candidate_ids": question.candidate_ids,
                "logits": [round(value, 8) for value in selected],
                "probabilities": [round(value, 8) for value in probabilities],
                "decode_steps": 0,
                "mode": "logits",
                "forward_seconds": round(elapsed, 8),
                "prompt_version": PROMPT_VERSION,
            }
            if question.qid in request.gold:
                record["gold"] = request.gold[question.qid]
                gold = request.gold[question.qid]
                if question.kind == BOOLEAN:
                    index = 1 if gold is True or gold in (1, "true", "True") else 0
                elif question.kind == SCORE:
                    index = int(gold)
                else:
                    index = question.candidate_ids.index(gold)
                record["correct"] = max(range(len(probabilities)), key=probabilities.__getitem__) == index
            if question.qid in request.teacher:
                record["teacher"] = request.teacher[question.qid]
            records.append(record)
    return records
