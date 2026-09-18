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
    evaluate.add_argument("--device", default="auto")

    demo = subparsers.add_parser("demo", help="score synthetic decisions and write a replay bundle")
    demo.add_argument("--checkpoint", required=True)
    demo.add_argument("--output", required=True)
    demo.add_argument("--count", type=int, default=12)
    demo.add_argument("--seed", type=int, default=17)
    demo.add_argument("--device", default="auto")

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
    print(json.dumps({"checkpoint": str(Path(args.output_dir) / "checkpoint"), **summary}))
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
    records, summary = evaluate_requests(model, requests, mode=args.mode)
    if args.predictions:
        write_jsonl(records, args.predictions)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _cmd_demo(args) -> int:
    model = DecisionModel.load(args.checkpoint, device=args.device)
    requests = generate_split(args.count, seed=args.seed, split="test")
    records = model.score(requests, ScoreOptions(mode="fresh", device=args.device))
    summary = aggregate(records)
    bundle = {
        "model": {
            "checkpoint": str(args.checkpoint),
            "config": model.config,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
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
        "info": _cmd_info,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
