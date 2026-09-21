"""Command line interface: synth, train, score, evaluate, demo, info."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .data import read_requests, write_jsonl, write_requests
from .metrics import aggregate
from .model import DecisionModel, ScoreOptions
from .synth import generate_split
from .train import TrainConfig, evaluate_requests, train


def _add_synth_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out", required=True)


def _add_backbone_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--intermediate-size", type=int, default=384)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="minojev", description="Generation-free decision model")
    parser.add_argument("--version", action="version", version=f"minojev {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    synth = subparsers.add_parser("synth", help="generate a synthetic decision split as JSONL")
    _add_synth_arguments(synth)
    synth.add_argument("--split", choices=["train", "dev", "test"], default="train")

    train_parser = subparsers.add_parser("train", help="train decision heads and save a checkpoint")
    train_parser.add_argument("--train", help="training requests JSONL; omit to generate synthetic data")
    train_parser.add_argument("--dev", help="dev requests JSONL; omit to generate synthetic data")
    train_parser.add_argument("--output-dir", required=True)
    train_parser.add_argument("--steps", type=int, default=300)
    train_parser.add_argument("--head-steps", type=int, default=25)
    train_parser.add_argument("--batch-requests", type=int, default=8)
    train_parser.add_argument("--eval-every", type=int, default=50)
    train_parser.add_argument("--learning-rate", type=float, default=2e-4)
    train_parser.add_argument("--head-lr", type=float, default=1e-3)
    train_parser.add_argument("--objective", choices=["teacher_ce", "gold_ce", "brier"], default="teacher_ce")
    train_parser.add_argument(
        "--calibrate",
        choices=["gold", "teacher", "none"],
        default="gold",
        help="fit probability temperatures on the dev set after training",
    )
    train_parser.add_argument("--train-count", type=int, default=256)
    train_parser.add_argument("--dev-count", type=int, default=64)
    train_parser.add_argument("--seed", type=int, default=17)
    train_parser.add_argument("--device", default="auto")
    _add_backbone_arguments(train_parser)

    score = subparsers.add_parser("score", help="score decision requests with a checkpoint")
    score.add_argument("--checkpoint", required=True)
    score.add_argument("--input", required=True)
    score.add_argument("--output", required=True)
    score.add_argument("--mode", choices=["fresh", "reuse"], default="fresh")
    score.add_argument("--batch-requests", type=int, default=16)
    score.add_argument("--device", default="auto")

    evaluate = subparsers.add_parser("evaluate", help="score requests and report aggregate metrics")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--input", required=True)
    evaluate.add_argument("--predictions", help="optional JSONL destination for per-question records")
    evaluate.add_argument("--mode", choices=["fresh", "reuse"], default="fresh")
    evaluate.add_argument("--limit", type=int, default=0, help="only evaluate the first N requests (0 = all)")
    evaluate.add_argument("--balanced", action="store_true", help="round-robin the limit across sources")
    evaluate.add_argument("--batch-requests", type=int, default=8, help="requests per forward pass (memory control)")
    evaluate.add_argument("--device", default="auto")

    demo = subparsers.add_parser("demo", help="score decisions and write a replay bundle")
    demo.add_argument("--checkpoint", required=True)
    demo.add_argument("--output", required=True)
    demo.add_argument("--input", help="requests JSONL to score; defaults to synthetic attribute test data")
    demo.add_argument("--limit", type=int, default=0, help="only score the first N requests from --input")
    demo.add_argument("--count", type=int, default=12)
    demo.add_argument("--seed", type=int, default=17)
    demo.add_argument("--device", default="auto")

    compare = subparsers.add_parser("compare", help="benchmark zero-decoding decisions against a generative baseline")
    compare.add_argument("--checkpoint", required=True)
    compare.add_argument("--input", required=True)
    compare.add_argument("--limit", type=int, default=120, help="number of requests to compare")
    compare.add_argument("--generative-model", default=None, help="defaults to the checkpoint backbone source")
    compare.add_argument("--max-new-tokens", type=int, default=12)
    compare.add_argument(
        "--generative-dtype",
        choices=["float32", "bfloat16"],
        default="bfloat16",
        help="precision for the generative baseline (bfloat16 halves memory on Apple Silicon)",
    )
    compare.add_argument("--mode", choices=["fresh", "reuse"], default="fresh")
    compare.add_argument("--output", default="results/compare.json")
    compare.add_argument("--markdown", default="results/compare.md")
    compare.add_argument("--chat-template", action="store_true", help="wrap prompts in the tokenizer chat template")
    compare.add_argument("--device", default="auto")

    watch = subparsers.add_parser("watch", help="live training dashboard for a run directory")
    watch.add_argument("--run", required=True, help="run directory containing status.json / metrics.jsonl")
    watch.add_argument("--host", default="127.0.0.1")
    watch.add_argument("--port", type=int, default=8010)

    serve = subparsers.add_parser("serve", help="serve a checkpoint over a local HTTP decision API")
    serve.add_argument("--checkpoint", required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--device", default="auto")

    maze_data = subparsers.add_parser("maze-data", help="generate grid-world decision requests as JSONL")
    maze_data.add_argument("--out", required=True)
    maze_data.add_argument("--count", type=int, default=512)
    maze_data.add_argument("--split", choices=["train", "dev", "test"], default="train")
    maze_data.add_argument("--seed", type=int, default=17)
    maze_data.add_argument("--size", type=int, default=6)

    maze_rollout = subparsers.add_parser("maze-rollout", help="run the maze controller and export replay frames")
    maze_rollout.add_argument("--checkpoint", required=True)
    maze_rollout.add_argument("--output", required=True)
    maze_rollout.add_argument("--count", type=int, default=6)
    maze_rollout.add_argument("--seed", type=int, default=23)
    maze_rollout.add_argument("--size", type=int, default=6)
    maze_rollout.add_argument("--device", default="auto")

    snake_data = subparsers.add_parser("snake-data", help="generate snake decision requests as JSONL")
    snake_data.add_argument("--out", required=True)
    snake_data.add_argument("--count", type=int, default=512)
    snake_data.add_argument("--split", choices=["train", "dev", "test"], default="train")
    snake_data.add_argument("--seed", type=int, default=17)
    snake_data.add_argument("--size", type=int, default=6)
    snake_data.add_argument("--length", type=int, default=3)

    snake_rollout = subparsers.add_parser("snake-rollout", help="run the snake controller and export replay frames")
    snake_rollout.add_argument("--checkpoint", required=True)
    snake_rollout.add_argument("--output", required=True)
    snake_rollout.add_argument("--count", type=int, default=6)
    snake_rollout.add_argument("--seed", type=int, default=7)
    snake_rollout.add_argument("--size", type=int, default=6)
    snake_rollout.add_argument("--length", type=int, default=3)
    snake_rollout.add_argument("--max-steps", type=int, default=120)
    snake_rollout.add_argument("--target-food", type=int, default=5)
    snake_rollout.add_argument("--device", default="auto")

    sokoban_data = subparsers.add_parser("sokoban-data", help="generate Sokoban decision requests as JSONL")
    sokoban_data.add_argument("--out", required=True)
    sokoban_data.add_argument("--count", type=int, default=512)
    sokoban_data.add_argument("--split", choices=["train", "dev", "test"], default="train")
    sokoban_data.add_argument("--seed", type=int, default=17)
    sokoban_data.add_argument("--size", type=int, default=7)
    sokoban_data.add_argument("--boxes", type=int, default=1)

    sokoban_rollout = subparsers.add_parser("sokoban-rollout", help="run the Sokoban controller and export replay frames")
    sokoban_rollout.add_argument("--checkpoint", required=True)
    sokoban_rollout.add_argument("--output", required=True)
    sokoban_rollout.add_argument("--count", type=int, default=8)
    sokoban_rollout.add_argument("--seed", type=int, default=11)
    sokoban_rollout.add_argument("--size", type=int, default=7)
    sokoban_rollout.add_argument("--boxes", type=int, default=1)
    sokoban_rollout.add_argument("--max-steps", type=int, default=90)
    sokoban_rollout.add_argument("--device", default="auto")

    build_data = subparsers.add_parser("build-data", help="convert public datasets into general decision JSONL splits")
    build_data.add_argument("--out-dir", default="data")
    build_data.add_argument("--per-source", type=int, default=4000, help="rows per training source")
    build_data.add_argument("--per-ood", type=int, default=1000, help="rows per held-out evaluation source")
    build_data.add_argument("--seed", type=int, default=17)
    build_data.add_argument("--sources", default=None, help="comma-separated registry keys; defaults to all")

    domain_data = subparsers.add_parser("domain-data", help="generate multi-domain decision requests as JSONL")
    domain_data.add_argument("--out", required=True)
    domain_data.add_argument("--count", type=int, default=512)
    domain_data.add_argument("--split", choices=["train", "dev", "test"], default="train")
    domain_data.add_argument("--seed", type=int, default=17)
    domain_data.add_argument("--domain", default=None, help="one of support/moderation/code-review/refunds/triage/leads")

    posttrain = subparsers.add_parser("posttrain", help="post-train decision heads on a pretrained HF backbone")
    posttrain.add_argument("--backbone", default="Qwen/Qwen3-0.6B-Base")
    posttrain.add_argument("--mode", choices=["head", "lora", "last-layers"], default="lora")
    posttrain.add_argument("--train", required=True)
    posttrain.add_argument("--dev", required=True)
    posttrain.add_argument("--output-dir", required=True)
    posttrain.add_argument("--steps", type=int, default=400)
    posttrain.add_argument("--head-steps", type=int, default=60)
    posttrain.add_argument("--batch-requests", type=int, default=2)
    posttrain.add_argument("--cache-batch-requests", type=int, default=2)
    posttrain.add_argument("--eval-every", type=int, default=100)
    posttrain.add_argument("--learning-rate", type=float, default=2e-4)
    posttrain.add_argument("--head-lr", type=float, default=1e-3)
    posttrain.add_argument("--lora-r", type=int, default=8)
    posttrain.add_argument("--lora-alpha", type=int, default=16)
    posttrain.add_argument("--last-layers", type=int, default=4)
    posttrain.add_argument("--objective", choices=["teacher_ce", "gold_ce", "brier"], default="teacher_ce")
    posttrain.add_argument("--seed", type=int, default=17)
    posttrain.add_argument("--device", default="auto")
    posttrain.add_argument("--no-save-backbone", action="store_true")
    posttrain.add_argument(
        "--max-memory-gb",
        type=float,
        default=12.0,
        help="abort a step when MPS driver or CPU peak RSS exceeds this budget; 0 disables the guard",
    )
    posttrain.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="trade compute for activation memory when training the backbone",
    )
    posttrain.add_argument("--log-every", type=int, default=10, help="monitor status interval in steps")
    posttrain.add_argument(
        "--inference-dtype",
        choices=["float32", "bfloat16"],
        default="float32",
        help="head mode only: bfloat16 speeds up feature caching on MPS",
    )

    calibrate = subparsers.add_parser("calibrate", help="fit probability temperatures on a dev set and save a checkpoint")
    calibrate.add_argument("--checkpoint", required=True)
    calibrate.add_argument("--input", required=True)
    calibrate.add_argument("--output", required=True)
    calibrate.add_argument("--mode", choices=["fresh", "reuse"], default="fresh")
    calibrate.add_argument("--target", choices=["gold", "teacher"], default="gold")
    calibrate.add_argument(
        "--from-records",
        action="store_true",
        help="fit temperatures from scored dev records instead of a second forward pass",
    )
    calibrate.add_argument("--device", default="auto")

    bench = subparsers.add_parser("bench", help="measure latency and throughput for each serving mode")
    bench.add_argument("--checkpoint", required=True)
    bench.add_argument("--input", required=True)
    bench.add_argument("--output", required=True)
    bench.add_argument("--modes", default="fresh,reuse")
    bench.add_argument("--repeats", type=int, default=3)
    bench.add_argument("--max-latency-requests", type=int, default=64)
    bench.add_argument("--batch-requests", type=int, default=16)
    bench.add_argument("--device", default="auto")

    info = subparsers.add_parser("info", help="print checkpoint metadata")
    info.add_argument("--checkpoint", required=True)
    return parser


def _cmd_synth(args) -> int:
    requests = generate_split(args.count, seed=args.seed, split=args.split)
    write_requests(requests, args.out)
    print(json.dumps({"split": args.split, "requests": len(requests), "out": args.out}))
    return 0


def _cmd_train(args) -> int:
    if args.train:
        train_requests = read_requests(args.train)
    else:
        train_requests = generate_split(args.train_count, seed=args.seed, split="train")
    if args.dev:
        dev_requests = read_requests(args.dev)
    else:
        dev_requests = generate_split(args.dev_count, seed=args.seed, split="dev")
    model = DecisionModel.new(
        hidden_size=args.hidden_size,
        num_layers=args.layers,
        num_heads=args.heads,
        intermediate_size=args.intermediate_size,
    )
    config = TrainConfig(
        steps=args.steps,
        head_steps=args.head_steps,
        batch_requests=args.batch_requests,
        eval_every=args.eval_every,
        learning_rate=args.learning_rate,
        head_lr=args.head_lr,
        objective=args.objective,
        seed=args.seed,
        device=args.device,
    )
    summary = train(model, train_requests, dev_requests, config, args.output_dir)
    checkpoint = str(Path(args.output_dir) / "checkpoint")
    if args.calibrate != "none":
        from .calibrate import fit_calibration

        best = DecisionModel.load(checkpoint, device=args.device)
        calibration, report = fit_calibration(best, dev_requests, target=args.calibrate, device=args.device)
        best.calibration = calibration
        best.config["calibration_report"] = report
        best.save(checkpoint)
        summary["calibration"] = report
        summary_path = Path(args.output_dir) / "summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"checkpoint": checkpoint, **summary}))
    return 0


def _cmd_score(args) -> int:
    model = DecisionModel.load(args.checkpoint, device=args.device)
    requests = read_requests(args.input)
    records = model.score(
        requests,
        ScoreOptions(mode=args.mode, batch_requests=args.batch_requests, device=args.device),
    )
    write_jsonl(records, args.output)
    print(json.dumps({"records": len(records), "output": args.output, "mode": args.mode, "decode_steps": 0}))
    return 0


def _cmd_evaluate(args) -> int:
    model = DecisionModel.load(args.checkpoint, device=args.device)
    requests = read_requests(args.input)
    if args.limit:
        if args.balanced:
            from .compare import select_balanced

            requests = select_balanced(requests, args.limit)
        else:
            requests = requests[: args.limit]
    records, summary = evaluate_requests(model, requests, mode=args.mode, device=args.device, batch_requests=args.batch_requests)
    if args.predictions:
        write_jsonl(records, args.predictions)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _cmd_demo(args) -> int:
    model = DecisionModel.load(args.checkpoint, device=args.device)
    if args.input:
        requests = read_requests(args.input)
        if args.limit:
            requests = requests[: args.limit]
    else:
        requests = generate_split(args.count, seed=args.seed, split="test")
    records = model.score(requests, ScoreOptions(mode="fresh", device=args.device))
    summary = aggregate(records)
    bundle = {
        "model": {
            "checkpoint": str(args.checkpoint),
            "config": model.config,
            "parameters": sum(parameter.numel() for parameter in model.trainable_parameters()),
        },
        "summary": summary,
        "records": records,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": args.output, "records": len(records), "summary": summary}, ensure_ascii=False, indent=2))
    return 0


def _cmd_maze_data(args) -> int:
    from .maze import generate_split as generate_maze_split

    requests = generate_maze_split(args.count, seed=args.seed, split=args.split, size=args.size)
    write_requests(requests, args.out)
    print(json.dumps({"split": args.split, "requests": len(requests), "size": args.size, "out": args.out}))
    return 0


def _cmd_maze_rollout(args) -> int:
    from .maze import rollout_bundle

    model = DecisionModel.load(args.checkpoint, device=args.device)
    bundle = rollout_bundle(model, count=args.count, seed=args.seed, size=args.size, device=args.device)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": args.output, **bundle["summary"]}))
    return 0


def _cmd_snake_data(args) -> int:
    from .snake import generate_split as generate_snake_split

    requests = generate_snake_split(args.count, seed=args.seed, split=args.split, size=args.size, length=args.length)
    write_requests(requests, args.out)
    print(json.dumps({"split": args.split, "requests": len(requests), "size": args.size, "out": args.out}))
    return 0


def _cmd_snake_rollout(args) -> int:
    from .snake import rollout_bundle

    model = DecisionModel.load(args.checkpoint, device=args.device)
    bundle = rollout_bundle(
        model,
        count=args.count,
        seed=args.seed,
        size=args.size,
        length=args.length,
        max_steps=args.max_steps,
        target_food=args.target_food,
        device=args.device,
    )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": args.output, **bundle["summary"]}))
    return 0


def _cmd_sokoban_data(args) -> int:
    from .sokoban import generate_split as generate_sokoban_split

    requests = generate_sokoban_split(args.count, seed=args.seed, split=args.split, size=args.size, boxes=args.boxes)
    write_requests(requests, args.out)
    print(json.dumps({"split": args.split, "requests": len(requests), "size": args.size, "out": args.out}))
    return 0


def _cmd_sokoban_rollout(args) -> int:
    from .sokoban import rollout_bundle

    model = DecisionModel.load(args.checkpoint, device=args.device)
    bundle = rollout_bundle(
        model,
        count=args.count,
        seed=args.seed,
        size=args.size,
        boxes=args.boxes,
        max_steps=args.max_steps,
        device=args.device,
    )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": args.output, **bundle["summary"]}))
    return 0


def _cmd_build_data(args) -> int:
    from .dataset_build import BuildConfig, build_dataset

    sources = [name.strip() for name in args.sources.split(",")] if args.sources else None
    result = build_dataset(
        BuildConfig(
            out_dir=args.out_dir,
            per_source=args.per_source,
            per_ood=args.per_ood,
            seed=args.seed,
            sources=sources,
        )
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


def _cmd_domain_data(args) -> int:
    from .domains import generate_split as generate_domain_split

    requests = generate_domain_split(args.count, seed=args.seed, split=args.split, domain=args.domain)
    write_requests(requests, args.out)
    print(json.dumps({"split": args.split, "domain": args.domain or "mixed", "requests": len(requests), "out": args.out}))
    return 0


def _cmd_posttrain(args) -> int:
    from .posttrain import PostTrainConfig, build_model, posttrain

    train_requests = read_requests(args.train)
    dev_requests = read_requests(args.dev)
    config = PostTrainConfig(
        backbone=args.backbone,
        mode=args.mode,
        steps=args.steps,
        head_steps=args.head_steps,
        batch_requests=args.batch_requests,
        cache_batch_requests=args.cache_batch_requests,
        eval_every=args.eval_every,
        learning_rate=args.learning_rate,
        head_lr=args.head_lr,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        last_layers=args.last_layers,
        objective=args.objective,
        seed=args.seed,
        device=args.device,
        save_backbone=not args.no_save_backbone,
        max_memory_gb=args.max_memory_gb or None,
        gradient_checkpointing=args.gradient_checkpointing,
        log_every=args.log_every,
        inference_dtype=args.inference_dtype,
    )
    model = build_model(config)
    summary = posttrain(model, train_requests, dev_requests, config, args.output_dir)
    print(json.dumps({"checkpoint": str(Path(args.output_dir) / "checkpoint"), **summary}, ensure_ascii=False))
    return 0


def _cmd_calibrate(args) -> int:
    from .calibrate import fit_calibration

    model = DecisionModel.load(args.checkpoint, device=args.device)
    requests = read_requests(args.input)
    if args.from_records:
        from .calibrate import fit_calibration_from_records

        records = model.score(requests, ScoreOptions(mode=args.mode, device=args.device))
        calibration, report = fit_calibration_from_records(records, target=args.target)
    else:
        calibration, report = fit_calibration(model, requests, mode=args.mode, target=args.target, device=args.device)
    model.calibration = calibration
    model.config["calibration_report"] = report
    model.save(args.output)
    print(json.dumps({"checkpoint": args.output, **report}, ensure_ascii=False, indent=2))
    return 0


def _cmd_bench(args) -> int:
    from .bench import BenchOptions, benchmark_model, compare_modes

    model = DecisionModel.load(args.checkpoint, device=args.device)
    requests = read_requests(args.input)
    result = benchmark_model(
        model,
        requests,
        BenchOptions(
            modes=tuple(mode.strip() for mode in args.modes.split(",") if mode.strip()),
            repeats=args.repeats,
            max_latency_requests=args.max_latency_requests,
            batch_requests=args.batch_requests,
            device=args.device,
        ),
    )
    result["comparison"] = compare_modes(result)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": args.output, **result}, ensure_ascii=False, indent=2))
    return 0


def _cmd_serve(args) -> int:
    from .serve import serve

    serve(args.checkpoint, host=args.host, port=args.port, device=args.device)
    return 0


def _cmd_compare(args) -> int:
    from .backbone import resolve_device
    from .compare import CompareOptions, compare_workload, write_report

    model = DecisionModel.load(args.checkpoint, device=args.device)
    requests = read_requests(args.input)
    backbone = model.config.get("backbone", {})
    source = args.generative_model or backbone.get("source") or "Qwen/Qwen3-1.7B"

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(source)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    generative_dtype = torch.bfloat16 if args.generative_dtype == "bfloat16" else torch.float32
    generative_model = AutoModelForCausalLM.from_pretrained(source, dtype=generative_dtype)
    generative_model.to(resolve_device(args.device))
    generative_model.eval()

    report = compare_workload(
        model,
        requests,
        generative_model,
        tokenizer,
        CompareOptions(
            limit=args.limit,
            mode=args.mode,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            chat_template=args.chat_template,
        ),
    )
    report["generative_model"] = source
    write_report(report, args.output, args.markdown)
    summary = {key: report[key] for key in ("requests", "decisions", "engine", "generative")}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _cmd_watch(args) -> int:
    from .watch import watch

    watch(args.run, host=args.host, port=args.port)
    return 0


def _cmd_info(args) -> int:
    config = json.loads((Path(args.checkpoint) / "config.json").read_text())
    print(json.dumps(config, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {
        "synth": _cmd_synth,
        "train": _cmd_train,
        "score": _cmd_score,
        "evaluate": _cmd_evaluate,
        "demo": _cmd_demo,
        "maze-data": _cmd_maze_data,
        "maze-rollout": _cmd_maze_rollout,
        "snake-data": _cmd_snake_data,
        "snake-rollout": _cmd_snake_rollout,
        "sokoban-data": _cmd_sokoban_data,
        "sokoban-rollout": _cmd_sokoban_rollout,
        "domain-data": _cmd_domain_data,
        "build-data": _cmd_build_data,
        "posttrain": _cmd_posttrain,
        "calibrate": _cmd_calibrate,
        "bench": _cmd_bench,
        "serve": _cmd_serve,
        "watch": _cmd_watch,
        "compare": _cmd_compare,
        "info": _cmd_info,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
