# minojev

**A minimal generation-free decision model.** States and questions in, complete
probability distributions out — with zero output-token decoding.

minojev turns runtime-defined judgment problems into typed distributions in a
single forward pass. Instead of generating an answer sentence and parsing it
back, the model reads decisions directly from hidden states. The whole
pipeline — training data, decision head, serving, and evaluation — runs
offline on a laptop CPU.

## Features

- **Three decision primitives.** `choice` (2–255 described candidates),
  `boolean` (one proposition with optional true/false criteria), and `score`
  (2–10 ordered levels with an expected value).
- **Zero decoding.** Every record reports `decode_steps: 0`; decisions are
  probabilities, not sampled tokens.
- **Dynamic candidates.** The same head serves any candidate set supplied at
  request time, and it is permutation-equivariant.
- **Parallel judgment.** Independent questions over a shared state are scored
  together without cross-question attention.
- **Shared-state reuse.** Each distinct state is prefilled once; its KV cache
  is replicated across every question and candidate branch.
- **Trainable and auditable.** Cross-entropy, gold, and Brier objectives with
  dev-selected checkpoints; teacher targets are exact by construction.
- **Offline reproduction.** A built-in tiny transformer and byte tokenizer
  train from scratch in minutes, with no model downloads.
- **Native-logits engine.** Pretrained Hugging Face causal LMs can score
  declared options directly from next-token logits without any training.

## Live demos

- **[Maze agent replay](https://zeredy879.github.io/minojev/maze.html)** —
  an animated grid run where every step shows a choice distribution, four
  parallel safety booleans, and the code-enforced final move.
- **[Parallel decision console](https://zeredy879.github.io/minojev/console.html)**
  — one state, many runtime questions scored together with their teacher
  distributions.

Both pages are static and replay committed result bundles. To serve them
locally:

```bash
python3 -m http.server 8080 --bind 127.0.0.1 --directory web
# http://127.0.0.1:8080
```

## Install

```bash
uv venv --python 3.12 .venv && . .venv/bin/activate
uv pip install -e ".[test]"
# optional: Hugging Face backbones for the native-logits engine
uv pip install -e ".[hf]"
```

## Quick start

```bash
# 1. generate synthetic decision data (attribute lookup, comparison, support)
minojev synth --out data/train.jsonl --count 512 --seed 17 --split train
minojev synth --out data/dev.jsonl   --count 128 --seed 17 --split dev

# 2. train the tiny backbone and decision head from scratch
minojev train --train data/train.jsonl --dev data/dev.jsonl \
  --output-dir runs/synth --steps 800 --head-steps 30 --eval-every 100 --device cpu

# 3. score arbitrary requests
minojev score --checkpoint runs/synth/checkpoint --input examples/decisions.jsonl \
  --output results/example-scores.jsonl --mode reuse
```

Train the maze agent and export a replay bundle:

```bash
minojev maze-data --out data/maze-train.jsonl --count 1024 --seed 17 --split train
minojev train --train data/maze-train.jsonl --dev data/maze-dev.jsonl \
  --output-dir runs/maze --steps 1500 --head-steps 40 --eval-every 100 --device cpu
minojev maze-rollout --checkpoint runs/maze/checkpoint \
  --output web/data/maze.json --count 6 --seed 23
```

## Request format

```json
{
  "id": "route-1",
  "state": "Customer cannot access an account after a password reset.",
  "questions": {
    "queue": {
      "type": "choice",
      "instructions": "Which queue should handle this request?",
      "options": {"access": "Account access support.", "billing": "Billing support."}
    },
    "urgent": {
      "type": "boolean",
      "instructions": "Is the customer fully locked out?",
      "criteria": {"true": "No usable login path", "false": "Login path still works"}
    },
    "confidence": {
      "type": "score",
      "instructions": "How confident is this judgment?",
      "levels": ["very low", "low", "medium", "high", "very high"]
    }
  }
}
```

`state` may be text or JSON. Flat single-question rows
(`{"id", "state", "question", "options"}`) are accepted as choice questions.
Question ids identify responses and are never placed in the model input.
Optional `gold` and `teacher` maps enable evaluation and supervised training.

## Python API

```python
from minojev import DecisionModel, Request, make_choice_question, ScoreOptions

model = DecisionModel.load("runs/synth/checkpoint", device="cpu")
request = Request(
    id="r1",
    state={"color": "blue", "shape": "round"},
    questions=[make_choice_question("q", "Which value belongs to 'shape'?",
                                    {"round": "round", "blue": "blue", "tiny": "tiny"})],
)
record = model.score([request], ScoreOptions(mode="reuse"))[0]
print(record["candidate_ids"], record["probabilities"], record["decode_steps"])
```

## Native-logits engine (pretrained models)

For a pretrained Hugging Face causal LM, `minojev.logits` scores declared
options directly from next-token logits at fixed answer-slot tokens — the
zero-training route:

```python
from minojev import load_hf_backbone
from minojev.logits import score_logits
from minojev.data import read_requests

backbone = load_hf_backbone("Qwen/Qwen2.5-0.5B-Instruct")
records = score_logits(backbone, backbone.tokenizer, read_requests("examples/decisions.jsonl"))
```

The logits engine supports up to 16 letter slots; use the trained head engine
for larger choice sets.

## Repository layout

```
src/minojev/
  types.py      request validation and primitives
  encoding.py   candidate paths, shared state/suffix split
  backbone.py   TinyLM (RoPE + KV cache) and HF adapter
  heads.py      scalar scorer + set attention, primitive readouts
  model.py      score modes, checkpoints
  train.py      warmup, objectives, dev selection
  synth.py      deterministic attribute decision families
  maze.py       grid-world states, teachers, and agent rollout
  logits.py     native-logits readout
  metrics.py    accuracy, CE, distribution error, per-family splits
  cli.py        synth / train / score / evaluate / demo / maze commands
tests/          contract, equivalence, training, maze, CLI, logits tests
web/            landing page, decision console, maze replay
```

## Results

Both bundled models are 547k-parameter transformers trained from scratch on
CPU. "Teacher top-set" counts a decision as correct when the predicted
candidate is among the teacher's best (important for grid moves, where two
directions often tie). Regenerate everything with `scripts/build_results.sh`.

| Run | Questions | Accuracy | Teacher top-set | CE vs teacher | Dist. error |
|---|---:|---:|---:|---:|---:|
| Attribute decisions | 627 | 66.5% | 66.5% | 0.972 | 0.224 |
| Maze decisions | 1,536 | 83.8% | 89.5% | 0.536 | 0.113 |

Maze per-primitive top-set accuracy: **boolean safety 87.8%**, **distance
score 97.7%**, **move choice 87.9%**. The committed six-maze controller replay
solves **5/6 mazes in 128 steps** with 53 code-forced moves.

![Maze replay preview](https://zeredy879.github.io/minojev/data/maze-preview.svg)

Raw metrics and per-question predictions are committed under
[`results/`](results); replay bundles live in [`web/data/`](web/data).

## Tests

```bash
.venv/bin/python -m pytest tests -q
```

The suite covers validation limits, path encoding, permutation equivariance,
boolean and score readouts, fresh-vs-reuse equivalence, batch independence,
checkpoint round trips, an end-to-end training convergence check, the maze
task and rollout mechanics, the CLI, and the native-logits engine. It runs
offline on CPU in a few minutes.

## License

MIT — see [LICENSE](LICENSE).
