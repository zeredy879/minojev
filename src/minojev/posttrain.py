"""Post-training of decision heads on pretrained Hugging Face backbones.

Three modes:

- ``head``: freeze the backbone, cache readout hidden states once, then train
  only the decision head. Fastest, useful as a baseline.
- ``lora``: attach LoRA adapters to the backbone and train them with the head
  end to end. The practical quality setting for laptop hardware.
- ``last-layers``: fine-tune the last K transformer blocks plus the head.

All modes finish with dev-fitted temperature calibration and save a checkpoint
that :class:`~minojev.model.DecisionModel` can load directly.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import torch

from .backbone import load_hf_backbone
from .calibrate import fit_calibration
from .encoding import EncodedBatch, pad_paths
from .heads import DecisionHead, group_loss
from .metrics import aggregate
from .model import DecisionModel
from .tokenizer import HFTokenizerAdapter
from .train import _targets_for
from .types import Request

MODES = ("head", "lora", "last-layers")


@dataclass
class PostTrainConfig:
    backbone: str = "Qwen/Qwen3-0.6B-Base"
    mode: str = "lora"
    steps: int = 400
    head_steps: int = 60
    batch_requests: int = 2
    eval_every: int = 100
    head_lr: float = 1e-3
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    objective: str = "teacher_ce"
    seed: int = 17
    device: str = "auto"
    cache_batch_requests: int = 2
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_last_layers: int = 4
    last_layers: int = 4
    save_backbone: bool = True
    save_dtype: str = "bfloat16"
    max_memory_gb: float | None = 12.0
    gradient_checkpointing: bool = False
    log_every: int = 10
    inference_dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if self.objective not in ("teacher_ce", "gold_ce", "brier"):
            raise ValueError("objective must be teacher_ce, gold_ce, or brier")


def build_model(config: PostTrainConfig) -> DecisionModel:
    dtype = config.inference_dtype if config.mode == "head" else "float32"
    backbone = load_hf_backbone(config.backbone, dtype=dtype, device=config.device, for_training=True)
    if config.mode == "head":
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
    elif config.mode == "lora":
        _apply_lora(backbone, config)
    elif config.mode == "last-layers":
        _freeze_except_last_layers(backbone, config.last_layers)
    if backbone.tokenizer is None:
        raise RuntimeError(f"Could not load a tokenizer for {config.backbone}")
    if config.gradient_checkpointing and config.mode != "head":
        model_ref = backbone.model
        if hasattr(model_ref, "enable_input_require_grads"):
            model_ref.enable_input_require_grads()
        model_ref.gradient_checkpointing_enable()
    head = DecisionHead(backbone.hidden_size, attention_dim=256, num_heads=8)
    model = DecisionModel(backbone, backbone.tokenizer, head)
    from .backbone import resolve_device

    model.to(resolve_device(config.device))
    model.config["backbone_name"] = config.backbone
    model.config["posttrain_mode"] = config.mode
    return model


def _apply_lora(backbone, config: PostTrainConfig) -> None:
    from peft import LoraConfig, get_peft_model

    layers = getattr(backbone.base, "layers", [])
    if not layers:
        raise RuntimeError("This backbone does not expose transformer layers for LoRA")
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    present = {name for layer in layers for name, _ in layer.named_parameters()}
    targets = [name for name in targets if any(name in parameter for parameter in present)] or None
    lora = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        target_modules=targets or ["q_proj", "v_proj"],
    )
    backbone.model = get_peft_model(backbone.model, lora)
    backbone.source = config.backbone
    model = backbone.model
    model.print_trainable_parameters()


def _freeze_except_last_layers(backbone, count: int) -> None:
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    layers = getattr(backbone.base, "layers", [])
    if not layers:
        raise RuntimeError("This backbone does not expose transformer layers")
    for layer in list(layers)[-count:]:
        for parameter in layer.parameters():
            parameter.requires_grad_(True)
    for name, parameter in backbone.base.named_parameters():
        if name.endswith("norm.weight") or name.endswith("norm.bias"):
            parameter.requires_grad_(True)


@dataclass
class CachedRequest:
    encoded: EncodedBatch
    hidden: torch.Tensor
    targets: list[torch.Tensor]
    tokens: int = 0


@torch.no_grad()
def cache_features(
    model: DecisionModel,
    requests: list[Request],
    objective: str,
    batch_requests: int,
    device: str = "auto",
    label: str = "features",
    monitor=None,
) -> list[CachedRequest]:
    import sys
    import time

    model.eval_mode(device)
    target_device = next(model.head.parameters()).device
    items = []
    started = time.perf_counter()
    report_every = max(1, len(requests) // 10)
    for start in range(0, len(requests), batch_requests):
        subset = requests[start : start + batch_requests]
        encoded = model.encode(subset)
        tokens, mask = pad_paths(encoded.paths, model.tokenizer.pad_id, target_device)
        hidden, _ = model.backbone(input_ids=tokens, attention_mask=mask)
        lengths = mask.sum(-1) - 1
        rows = hidden[torch.arange(len(encoded.paths), device=target_device), lengths].float().cpu()
        targets = _targets_for(model, encoded, objective)
        tokens = sum(len(path.token_ids) for path in encoded.paths)
        items.append(CachedRequest(encoded=encoded, hidden=rows, targets=targets, tokens=tokens))
        done = start + len(subset)
        if done % report_every < batch_requests or done == len(requests):
            message = f"caching {label}: {done}/{len(requests)} requests ({round(time.perf_counter() - started, 1)}s)"
            print(f"[{label}] {message}", file=sys.stderr, flush=True)
            if monitor is not None:
                monitor.note(message)
    return items


def _merge_cached(items: list[CachedRequest]):
    groups = []
    kinds = []
    targets = []
    hidden_rows = []
    offset = 0
    request_count = 0
    for item in items:
        hidden_rows.append(item.hidden)
        for group, target in zip(item.encoded.groups, item.targets):
            groups.append(
                replace(
                    group,
                    request_index=group.request_index + request_count,
                    path_indices=[index + offset for index in group.path_indices],
                )
            )
            kinds.append(group.kind)
            targets.append(target)
        offset += item.hidden.shape[0]
        request_count += len(item.encoded.requests)
    return torch.cat(hidden_rows, dim=0), groups, kinds, targets


def evaluate_cached(model: DecisionModel, items: list[CachedRequest], objective: str = "teacher_ce") -> dict:
    device = next(model.head.parameters()).device
    records = []
    model.head.eval()
    with torch.no_grad():
        for item in items:
            hidden = item.hidden.to(device)
            outputs = model.head(hidden, item.encoded.groups, [group.kind for group in item.encoded.groups])
            records.extend(model._records(item.encoded, outputs, "cached", 0.0))
    return aggregate(records)


def train_head_cached(
    model: DecisionModel,
    train_requests: list[Request],
    dev_requests: list[Request],
    config: PostTrainConfig,
    output_dir: str | Path,
) -> dict:
    from .monitor import TrainingMonitor, release_memory

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = next(model.head.parameters()).device
    started = time.perf_counter()
    monitor = TrainingMonitor(
        output_dir,
        config.steps,
        log_every=config.log_every,
        max_memory_gb=config.max_memory_gb,
        label="head-cache",
    )
    monitor.note("caching training features")
    train_items = cache_features(
        model, train_requests, config.objective, config.cache_batch_requests, config.device, label="train", monitor=monitor
    )
    release_memory()
    monitor.note("caching dev features")
    dev_items = cache_features(
        model, dev_requests, config.objective, config.cache_batch_requests, config.device, label="dev", monitor=monitor
    )
    release_memory()
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=config.head_lr, weight_decay=config.weight_decay)
    log: list[dict] = []
    best_score = float("inf")
    best_step = None
    peak_memory = 0.0
    token_total = 0
    for step in range(config.steps):
        batch = random.sample(train_items, min(config.batch_requests, len(train_items)))
        hidden, groups, kinds, targets = _merge_cached(batch)
        token_total += sum(item.tokens for item in batch)
        model.head.train()
        optimizer.zero_grad(set_to_none=True)
        outputs = model.head(hidden.to(device), groups, kinds)
        inner = "brier" if config.objective == "brier" else "ce"
        loss = group_loss(outputs, groups, [target.to(device) for target in targets], inner).mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.head.parameters(), config.grad_clip))
        optimizer.step()
        monitor_entry = monitor.log(
            step + 1,
            "head",
            loss=float(loss.detach()),
            lr=config.head_lr,
            grad_norm=grad_norm,
            tokens=token_total,
        )
        if monitor_entry:
            peak_memory = max(
                peak_memory,
                monitor_entry["memory"].get("mps_driver_gb") or 0.0,
                monitor_entry["memory"].get("cpu_peak_rss_gb") or 0.0,
            )
        record = {
            "step": step + 1,
            "loss": round(float(loss.detach()), 6),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        if (step + 1) % config.eval_every == 0 or step + 1 == config.steps:
            metrics = evaluate_cached(model, dev_items, config.objective)
            record["dev"] = metrics
            score = metrics.get("teacher_ce" if config.objective != "gold_ce" else "gold_nll")
            if score is not None and score < best_score:
                best_score = float(score)
                best_step = step + 1
                _save_checkpoint(model, config, output_dir, metrics)
            monitor.log(
                step + 1,
                "head-eval",
                loss=float(loss.detach()),
                tokens=token_total,
                extra={"dev_accuracy": metrics.get("accuracy"), "dev_teacher_ce": metrics.get("teacher_ce")},
                force=True,
            )
        log.append(record)
    if best_step is None:
        raise RuntimeError("No dev evaluation selected a checkpoint")
    (output_dir / "train_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2) + "\n")
    summary = {
        "mode": config.mode,
        "backbone": config.backbone,
        "best_step": best_step,
        "best_dev_score": best_score,
        "train_requests": len(train_requests),
        "dev_requests": len(dev_requests),
        "training_seconds": round(time.perf_counter() - started, 3),
        "peak_memory_gb": round(peak_memory, 3),
        "max_memory_gb": config.max_memory_gb,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    model.config["posttrain"] = summary
    return summary


def train_end_to_end(
    model: DecisionModel,
    train_requests: list[Request],
    dev_requests: list[Request],
    config: PostTrainConfig,
    output_dir: str | Path,
) -> dict:
    from .train import TrainConfig, evaluate_requests, train

    output_dir = Path(output_dir)
    train_config = TrainConfig(
        steps=config.steps,
        head_steps=config.head_steps,
        batch_requests=config.batch_requests,
        eval_every=config.eval_every,
        head_lr=config.head_lr,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        grad_clip=config.grad_clip,
        objective=config.objective,
        seed=config.seed,
        device=config.device,
    )
    summary = train(model, train_requests, dev_requests, train_config, output_dir)
    summary["mode"] = config.mode
    summary["backbone"] = config.backbone
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def _save_checkpoint(model: DecisionModel, config: PostTrainConfig, output_dir: Path, metrics: dict) -> None:
    model.config["metrics"] = metrics
    model.save(output_dir / "checkpoint", save_backbone=config.mode != "head" and config.save_backbone)


def posttrain(
    model: DecisionModel,
    train_requests: list[Request],
    dev_requests: list[Request],
    config: PostTrainConfig,
    output_dir: str | Path,
) -> dict:
    from .monitor import TrainingMonitor, release_memory

    output_dir = Path(output_dir)
    calibration_mode = "teacher" if config.objective == "teacher_ce" else "gold"
    if config.mode == "head":
        summary = train_head_cached(model, train_requests, dev_requests, config, output_dir)
    else:
        summary = train_end_to_end(model, train_requests, dev_requests, config, output_dir)
    # Drop the training model before loading another copy for calibration; two
    # full models in unified memory is what previously exhausted the machine.
    del model
    release_memory()
    checkpoint = DecisionModel.load(output_dir / "checkpoint", device=config.device)
    if config.mode == "lora":
        _merge_adapters(checkpoint)
    monitor = TrainingMonitor(
        output_dir,
        config.steps,
        log_every=config.log_every,
        max_memory_gb=config.max_memory_gb,
        label="calibrate",
    )
    monitor.note("fitting calibration temperatures (chunked)")
    calibration, report = fit_calibration(
        checkpoint,
        dev_requests,
        target=calibration_mode,
        device=config.device,
        batch_requests=min(config.cache_batch_requests, 8),
        max_memory_gb=config.max_memory_gb,
    )
    release_memory()
    checkpoint.calibration = calibration
    checkpoint.config.setdefault("posttrain", summary)
    checkpoint.config["calibration_report"] = report
    summary["calibration"] = report
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    if config.mode == "head":
        checkpoint.config["backbone"]["dtype"] = "float32"
        checkpoint.save(output_dir / "checkpoint", save_backbone=False)
    else:
        if config.save_backbone and config.save_dtype == "bfloat16":
            _cast_backbone(checkpoint, torch.bfloat16)
            checkpoint.config["backbone"]["dtype"] = "bfloat16"
        checkpoint.save(output_dir / "checkpoint", save_backbone=config.save_backbone)
    return summary


def _merge_adapters(model: DecisionModel) -> None:
    backbone = model.backbone
    if hasattr(backbone.model, "merge_and_unload"):
        backbone.model = backbone.model.merge_and_unload()
        backbone.refresh_base()
        if hasattr(backbone.model, "config"):
            backbone.hidden_size = int(backbone.model.config.hidden_size)


def _cast_backbone(model: DecisionModel, dtype: torch.dtype) -> None:
    if hasattr(model.backbone, "model"):
        model.backbone.model.to(dtype)


def load_requests(path: str | Path) -> list[Request]:
    from .data import read_requests

    return read_requests(path)
