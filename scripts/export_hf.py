#!/usr/bin/env python3
"""Export calibrated checkpoints into a Hugging Face model repository layout."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

MODEL_CARD = """---
library_name: minojev
license: mit
language:
- en
- zh
tags:
- decision-model
- structured-output
- probability-calibration
- generation-free
- system-one
pipeline_tag: text-classification
---

# minojev

**Decisions, not tokens.** minojev turns runtime-defined judgment problems into
typed probability distributions in a single forward pass. No output token is
ever generated; decisions are read from hidden states.

This repository contains two small, fully reproducible checkpoints trained from
scratch (547k parameters each) on synthetic decision data. They are intended
for testing the pipeline, demos, and as a starting point for fine-tuning.

## Variants

| Folder | Task | Questions | Accuracy | Teacher top-set | ECE (calibrated) |
|---|---|---:|---:|---:|---:|
| `synth/` | Attribute decisions | 627 | {synth_accuracy} | {synth_topset} | {synth_ece} |
| `maze/` | Grid-world decisions | 1,536 | {maze_accuracy} | {maze_topset} | {maze_ece} |

The maze checkpoint drives the animated agent demo: 5/6 mazes solved, with
{bench_maze_reuse} decisions/s in shared-state reuse mode on a laptop CPU.

## Usage

```python
from huggingface_hub import snapshot_download
from minojev import DecisionModel

path = snapshot_download("{repo_id}", allow_patterns=["maze/*"])
model = DecisionModel.load(f"{{path}}/maze", device="cpu")

from minojev import Request, make_choice_question, ScoreOptions
request = Request(
    id="r1",
    state={{"size": 6, "agent": [0, 0], "goal": [5, 5], "step": 0, "walls": [[0, 1], [1, 3]]}},
    questions=[make_choice_question("move", "Which move reaches the goal?",
                                    {{d: f"move {{d}}" for d in ["up", "down", "left", "right"]}})],
)
record = model.score([request], ScoreOptions(mode="reuse"))[0]
print(record["candidate_ids"], record["probabilities"], record["decode_steps"])
```

## Decision primitives

- `choice`: 2-255 described candidates -> complete distribution.
- `boolean`: one proposition with optional true/false criteria -> `[P(false), P(true)]`.
- `score`: 2-10 ordered levels -> level distribution plus expected level.

Temperature scaling fitted on dev data calibrates confidence at serving time
without changing any predicted candidate.

## Training details

- Backbone: 96 hidden, 3 layers, 4 heads, RoPE positions, native KV cache.
- Tokenizer: byte-level (259 tokens), no external vocabulary.
- Objective: cross entropy to exact teacher distributions; head warmup then
  joint fine-tuning; dev-selected checkpoints.
- Calibration: per-primitive temperature minimizing dev negative log-likelihood.
- Hardware: CPU only, minutes per checkpoint.

## Links

- Code and demos: https://github.com/{owner}/minojev
- Interactive demo: https://{owner}.github.io/minojev/
- Datasets with teacher distributions: https://huggingface.co/datasets/{owner}/minojev-data
- English and Chinese READMEs: https://github.com/{owner}/minojev#readme

## License

MIT. The checkpoints are synthetic-task models; validate probabilities on your
own workload before using them in production.
"""


def metrics(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synth", default="runs/synth-calibrated")
    parser.add_argument("--maze", default="runs/maze-calibrated")
    parser.add_argument("--output", default="dist/hf")
    parser.add_argument("--repo-id", default="zeredy879/minojev")
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    for name, source in (("synth", args.synth), ("maze", args.maze)):
        shutil.copytree(Path(source), output / name)

    synth = metrics(Path("results/synth-metrics.json"))
    maze = metrics(Path("results/maze-metrics.json"))
    bench = metrics(Path("results/bench-maze.json"))
    owner = args.repo_id.split("/")[0]
    card = MODEL_CARD.format(
        repo_id=args.repo_id,
        owner=owner,
        synth_accuracy=f"{100 * synth['accuracy']:.1f}%",
        synth_topset=f"{100 * synth['teacher_topset_accuracy']:.1f}%",
        synth_ece=f"{synth['expected_calibration_error']:.3f}",
        maze_accuracy=f"{100 * maze['accuracy']:.1f}%",
        maze_topset=f"{100 * maze['teacher_topset_accuracy']:.1f}%",
        maze_ece=f"{maze['expected_calibration_error']:.3f}",
        bench_maze_reuse=f"{bench['modes']['reuse']['decisions_per_second']:.0f}",
    )
    (output / "README.md").write_text(card)
    (output / ".gitattributes").write_text(
        "*.safetensors filter=lfs diff=lfs merge=lfs -text\n"
        "*.bin filter=lfs diff=lfs merge=lfs -text\n"
    )
    print(json.dumps({"output": str(output), "variants": ["synth", "maze"], "repo_id": args.repo_id}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
