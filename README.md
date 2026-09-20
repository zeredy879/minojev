<p align="center"><img src="assets/banner.svg" alt="minojev — decisions, not tokens" width="960"></p>

<p align="center">
  <a href="README.zh-CN.md">简体中文</a> · <b>English</b> · <a href="https://zeredy879.github.io/minojev/">Live demos</a> · <a href="https://zeredy879.github.io/minojev/benchmark.html">Benchmark</a> · <a href="https://huggingface.co/zeredy879/minojev">Models</a> · <a href="https://huggingface.co/datasets/zeredy879/minojev-data">Data</a>
</p>

<p align="center">
  <a href="https://huggingface.co/zeredy879/minojev"><img alt="Hugging Face models" src="https://img.shields.io/badge/Hugging%20Face-models-f0b06a"></a>
  <a href="https://huggingface.co/datasets/zeredy879/minojev-data"><img alt="Hugging Face datasets" src="https://img.shields.io/badge/Hugging%20Face-datasets-59d3a8"></a>
  <img alt="license" src="https://img.shields.io/badge/license-MIT-f0b06a">
  <img alt="decode steps" src="https://img.shields.io/badge/decode__steps-0-6ea8fe">
  <img alt="head params" src="https://img.shields.io/badge/decision__head-0.8M-8d9bb3">
</p>

> **The one-sentence version.** By the time a model finishes reading your question it
> already has an opinion — minojev reads that opinion back as a typed, calibrated
> distribution instead of making the model write a sentence.

## Head training, in plain words

Most "Jev-style" projects either call a hosted API or fine-tune a whole model. minojev's
core is smaller than both:

1. **Freeze the backbone.** In the bundled release, Qwen3-1.7B never updates.
2. **Cache candidate features once.** Every candidate path (state + question + option) is
   forwarded a single time; the final hidden state is stored.
3. **Train only the decision head.** A shared scorer plus set attention — **~0.8M
   parameters** — is trained on those cached vectors with distribution losses.
4. **Calibrate on dev.** One temperature per primitive is fitted on held-out outcomes.

The whole run was **~37 minutes and ~4 GB of peak memory on an Apple Silicon laptop**.
Because the backbone is untouched, there is no catastrophic forgetting, no GPU required,
and every step is observable (see the [monitor dashboard](#memory-safe-training)).

## How it compares to token generation

Same backbone (Qwen3-1.7B), same questions, same prompts. The baseline must *generate*
its answer, token by token; minojev reads a typed distribution from hidden states.
Suite: 120 balanced decisions across banking77, CLINC150, Amazon Polarity, and GSM8K
verification. The generative baseline uses the chat template with thinking disabled and
is told to answer with a single letter.

| Metric | minojev (head-trained) | zero-shot logits readout | generative baseline |
|---|---:|---:|---:|
| Accuracy | **95.8%** | 88.3% | 80.0% |
| ECE | **0.024** | 0.097 | not available |
| Output tokens / decision | **0** | **0** | 3.48 |
| Time to first token (p50) | **0 ms** | **0 ms** | ~100 ms |
| Latency p50 | **264 ms** | — | 361 ms |
| Latency p95 | **~0.6 s** | — | ~3.3 s |
| Format / parse failures | **0** | 0 | 0 |

Full test suite (200 balanced requests, calibrated release): **97.5% accuracy, ECE
0.014**, mean confidence 0.963, and at confidence ≥ 0.9 the model covers **89.5% of
requests at 98.9% accuracy** — usable for accept/escalate policies.

**Honest boundary:** on a deliberate out-of-distribution workload (never-trained customer
support domains), the head-trained model scored 31.7% vs 40.0% for the zero-shot
generative baseline. Head training specializes; it is not a substitute for data breadth.
That is why the pipeline isolates OOD sources and reports them separately.

<img src="assets/explainer.svg" alt="A chat model generates tokens then parses text; minojev reads the distribution in one forward pass" width="100%">

## Try it in your browser

- **[Benchmark page](https://zeredy879.github.io/minojev/benchmark.html)** — accuracy,
  token economy, and latency side by side.
- **[Domain playground](https://zeredy879.github.io/minojev/playground.html)** — six
  operational domains answered as typed distributions.
- **[Maze agent](https://zeredy879.github.io/minojev/maze.html)** — a learned policy
  replayed step by step with its probabilities.
- **[Decision console](https://zeredy879.github.io/minojev/console.html)** — one state,
  many runtime questions.

## Quick start

```bash
uv venv --python 3.12 .venv && . .venv/bin/activate
uv pip install -e ".[test,monitor]"          # add .[hf] for Hugging Face backbones
```

Build the general dataset and run head training with live observability:

```bash
# 1. convert permissive public sources into decision requests (train/dev/test/OOD)
minojev build-data --per-source 2500 --per-ood 800

# 2. head training: frozen backbone, cached features, decision head only
minojev posttrain --backbone Qwen/Qwen3-1.7B --mode head \
  --train data/general-train.jsonl --dev data/general-dev.jsonl \
  --output-dir runs/general-head --steps 3000 \
  --inference-dtype bfloat16 --max-memory-gb 12

# 3. calibrate on dev outcomes
minojev calibrate --checkpoint runs/general-head/checkpoint \
  --input data/general-dev.jsonl --output runs/general-cal --target gold --from-records

# 4. compare against token generation with the same backbone
minojev compare --checkpoint runs/general-cal \
  --input data/general-test.jsonl --limit 120 --chat-template
```

Watch any run live from a second terminal:

```bash
minojev watch --run runs/general-head --port 8010   # http://127.0.0.1:8010/
```

## Watching a run (without cooking your laptop)

The monitor is not decoration: every training phase reports loss, gradient norm,
tokens/s, ETA, MPS driver memory, process RSS, CPU load, and system memory pressure to
`status.json` and `metrics.jsonl`. A hard budget (`--max-memory-gb`) aborts a step before
the machine is at risk, calibration forwards are chunked, and each phase frees the
previous model before loading the next. The 1.7B head run peaked at **4.0 GB** against a
12 GB budget.

## How we measure quality

Loss is a poor progress signal for a decision model — teacher distributions have an
entropy floor and heterogeneous batches make it noisy. The pipeline reports:

| Dimension | Metrics |
|---|---|
| Decision quality | accuracy, teacher top-set, per-source and per-candidate-count breakdown |
| Probability quality | ECE, Brier, gold NLL, selective accuracy (coverage @ confidence) |
| Token economy | input/output tokens per decision, decode steps (0) |
| Response speed | time to first token, per-request p50/p95, decisions/second, parse failures |

`minojev compare` produces JSON + Markdown reports from committed artifacts, and
`scripts/build_benchmark_bundle.py` feeds the live benchmark page.

## Models and data

- Checkpoints: [`zeredy879/minojev`](https://huggingface.co/zeredy879/minojev) —
  `general/` (Qwen3-1.7B + head), plus the tiny from-scratch `synth/` and `maze/` models.
- Data: [`zeredy879/minojev-data`](https://huggingface.co/datasets/zeredy879/minojev-data) —
  `general/{train,dev,test,ood}.jsonl` with source and license on every line.

```python
from huggingface_hub import snapshot_download
from minojev import DecisionModel

path = snapshot_download("zeredy879/minojev", allow_patterns=["general/*"])
model = DecisionModel.load(f"{path}/general", device="cpu")
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

`choice` accepts 2–255 candidates, `score` 2–10 ordered levels. Flat single-question rows
are accepted as choice questions. Question ids are never placed in the model input.

## Run it locally

```bash
minojev serve --checkpoint runs/general-cal --port 8000
curl -s localhost:8000/score -H 'content-type: application/json' \
  -d '{"id":"r1","state":"Where is my card?","question":"Which category?","options":{"card_arrival":"Card arrival","atm":"ATM"}}'
```

Records report `decode_steps: 0` and carry the full candidate distribution.

## Don't want to train at all?

For pretrained models, `minojev.logits` reads declared option logits directly at
letter slots — the zero-training route used in the benchmark:

```python
from minojev import load_hf_backbone
from minojev.logits import score_logits, score_logits_two_stage

backbone = load_hf_backbone("Qwen/Qwen3-1.7B")
records = score_logits(backbone, backbone.tokenizer, requests)
wide = score_logits_two_stage(backbone, backbone.tokenizer, requests)  # >16 candidates
```

## Repository layout

```
src/minojev/
  types.py        request validation and primitives
  encoding.py     candidate paths, state/suffix split
  backbone.py     TinyLM (RoPE + KV cache) and HF adapter
  heads.py        shared scorer + set attention, primitive readouts
  posttrain.py    head training / LoRA with cached features
  monitor.py      live status, memory budget, CPU/memory pressure
  watch.py        training dashboard server
  calibrate.py    temperature scaling (forward-based and record-based)
  generative.py   token-generating baseline with TTFT measurement
  compare.py      decision engine vs generation benchmark
  domains.py      six-domain synthetic decisions
  dataset_build.py converted public datasets with source-level splits
  logits.py       native-logits readout (single-pass and two-stage)
  serve.py        local HTTP decision API
  metrics.py      accuracy, ECE, Brier, selective accuracy, by-source
  maze.py         grid-world tasks and agent rollout
  synth.py        attribute decision families
tests/            offline suite (unit + end-to-end + dashboards)
web/              landing, benchmark, playground, maze, console (EN + 中文)
```

## Development notes

This project was built with AI assistance. Correctness is anchored by the offline test
suite, the committed per-question predictions and metrics, and the reproducible scripts
in [`scripts/`](scripts). Training data uses permissive licenses only (MIT, Apache-2.0,
CC-BY); NC-licensed datasets are reserved for evaluation.

## License

MIT — see [LICENSE](LICENSE). The `general/` checkpoint is a derivative of Qwen3-1.7B
(Apache-2.0) and inherits that license.
