"""Training loop for decision heads (and optionally the backbone).

The head is warmed up with the backbone frozen, then the backbone is
unfastened at a lower learning rate. Checkpoints are selected on dev loss
using the configured proper objective (cross entropy or Brier).
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from .metrics import aggregate
from .model import DecisionModel, ScoreOptions, gold_target
from .types import Request, teacher_distribution

OBJECTIVES = ("teacher_ce", "gold_ce", "brier")


@dataclass
class TrainConfig:
    steps: int = 300
    head_steps: int = 25
    batch_requests: int = 8
    eval_every: int = 50
    head_lr: float = 1e-3
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    objective: str = "teacher_ce"
    seed: int = 17
    device: str = "auto"
    backbone_params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}")
        if self.steps <= 0 or self.head_steps < 0:
            raise ValueError("steps must be positive and head_steps nonnegative")
        if self.batch_requests <= 0 or self.eval_every <= 0:
            raise ValueError("batch_requests and eval_every must be positive")


def _targets_for(model: DecisionModel, encoded, objective: str) -> list[torch.Tensor]:
    targets: list[torch.Tensor] = []
    for group, path_count in zip(encoded.groups, [len(group.path_indices) for group in encoded.groups]):
        request = encoded.requests[group.request_index]
        question = request.questions[group.question_index]
        if objective == "gold_ce":
            target = gold_target(question, request.gold.get(question.qid), path_count)
            if target is None:
                raise ValueError(f"Missing gold for {request.id}:{question.qid}")
        else:
            probabilities = teacher_distribution(question, request.teacher.get(question.qid))
            if probabilities is None:
                raise ValueError(f"Missing teacher distribution for {request.id}:{question.qid}")
            target = torch.tensor(probabilities, dtype=torch.float32)
        targets.append(target)
    return targets


def batch_loss(model: DecisionModel, encoded, objective: str) -> torch.Tensor:
    from .heads import group_loss

    hidden = model.forward_paths(encoded)
    outputs = model.head(hidden, encoded.groups, [group.kind for group in encoded.groups])
    targets = _targets_for(model, encoded, objective)
    inner = "brier" if objective == "brier" else "ce"
    return group_loss(outputs, encoded.groups, targets, inner).mean()


def evaluate_requests(
    model: DecisionModel,
    requests: list[Request],
    mode: str = "fresh",
    device: str = "auto",
    batch_requests: int = 8,
) -> tuple[list[dict], dict]:
    records = model.score(requests, ScoreOptions(mode=mode, batch_requests=batch_requests, device=device))
    return records, aggregate(records)


def train(
    model: DecisionModel,
    train_requests: list[Request],
    dev_requests: list[Request],
    config: TrainConfig,
    output_dir: str | Path,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    model.eval_mode(config.device)
    device = next(model.head.parameters()).device

    encoded_train = [model.encode([request]) for request in train_requests]
    head_parameters = [parameter for parameter in model.head.parameters()]
    backbone_parameters = model.backbone_parameters()
    trainable_flags = [parameter.requires_grad for parameter in backbone_parameters]
    trainable_backbone = [parameter for parameter, flag in zip(backbone_parameters, trainable_flags) if flag]
    model.train_mode()
    for parameter in backbone_parameters:
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [
            {"params": trainable_backbone, "lr": config.learning_rate},
            {"params": head_parameters, "lr": config.head_lr},
        ],
        weight_decay=config.weight_decay,
    )
    log: list[dict] = []
    best_score = float("inf")
    best_step = None
    started = time.perf_counter()
    for step in range(config.steps):
        warmup = step < config.head_steps
        optimizer.param_groups[1]["lr"] = config.head_lr if warmup else config.learning_rate
        for parameter, flag in zip(backbone_parameters, trainable_flags):
            parameter.requires_grad_(flag and not warmup)
        batch_indices = random.sample(range(len(encoded_train)), min(config.batch_requests, len(encoded_train)))
        batch = _merge_encoded([encoded_train[index] for index in batch_indices])
        model.train_mode()
        optimizer.zero_grad(set_to_none=True)
        loss = batch_loss(model, batch, config.objective)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), config.grad_clip)
        optimizer.step()
        entry = {
            "step": step + 1,
            "phase": "head" if warmup else "full",
            "loss": round(float(loss.detach()), 6),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        selected = None
        if not warmup and ((step + 1 - config.head_steps) % config.eval_every == 0 or step + 1 == config.steps):
            _, dev_metrics = evaluate_requests(model, dev_requests, device=config.device)
            entry["dev"] = dev_metrics
            selected = dev_metrics.get("teacher_ce" if config.objective != "gold_ce" else "gold_nll")
            if selected is None:
                selected = dev_metrics.get("distribution_error")
            if selected is not None and selected < best_score:
                best_score = float(selected)
                best_step = step + 1
                model.config["objective"] = config.objective
                model.config["metrics"] = dev_metrics
                model.save(output_dir / "checkpoint")
        log.append(entry)
    if best_step is None:
        raise RuntimeError("No dev evaluation selected a checkpoint; increase steps or lower head_steps")
    (output_dir / "train_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2) + "\n")
    (output_dir / "train_config.json").write_text(json.dumps(asdict(config), ensure_ascii=False, indent=2) + "\n")
    summary = {
        "best_step": best_step,
        "best_dev_score": best_score,
        "objective": config.objective,
        "train_requests": len(train_requests),
        "dev_requests": len(dev_requests),
        "training_seconds": round(time.perf_counter() - started, 3),
        "parameter_count": sum(parameter.numel() for parameter in model.trainable_parameters()),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    model.config["objective"] = config.objective
    return summary


def _merge_encoded(batches):
    from dataclasses import replace

    from .encoding import EncodedBatch

    requests = []
    paths = []
    groups = []
    state_keys = []
    for batch in batches:
        offset = len(paths)
        base = len(requests)
        requests.extend(batch.requests)
        for path in batch.paths:
            paths.append(replace(path, request_index=path.request_index + base))
        for group in batch.groups:
            groups.append(
                replace(
                    group,
                    request_index=group.request_index + base,
                    path_indices=[index + offset for index in group.path_indices],
                )
            )
        state_keys.extend(batch.state_keys)
    return EncodedBatch(requests=requests, paths=paths, groups=groups, state_keys=state_keys)
