#!/usr/bin/env python3
"""Export the general decision model (Qwen3-1.7B + trained head) for the Hub."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

MODEL_CARD = """---
library_name: minojev
license: apache-2.0
base_model: Qwen/Qwen3-1.7B
language:
- en
tags:
- decision-model
- structured-output
- probability-calibration
- generation-free
- head-training
- system-one
pipeline_tag: text-classification
---

# minojev general decision model (Qwen3-1.7B + trained head)

**Decisions, not tokens.** This checkpoint turns Qwen3-1.7B into a typed decision model:
it returns calibrated probability distributions for `Choice`, `Boolean`, and `Score`
questions in one forward pass, with **zero output tokens**. It was produced by
**head training**: the backbone stays frozen, candidate-path features are cached once,
and only the decision head is trained, followed by dev-fitted temperature calibration.

## Results

Test suite: 200 requests, balanced across four sources (banking77, CLINC150, Amazon
Polarity, GSM8K verification); every candidate is declared in the request.

| Metric | Value |
|---|---|
| Accuracy | **97.5%** |
| Expected calibration error (ECE) | **0.014** |
| Mean confidence | 0.963 |
| Selective accuracy (confidence >= 0.9) | 89.5% coverage at **98.9%** accuracy |
| Output tokens per decision | **0** |

Head-to-head against token generation with the same backbone and prompts
(120-decision balanced suite, chat template, thinking disabled):

| Metric | minojev | generative baseline |
|---|---:|---:|
| Accuracy | **95.8%** | 80.0% |
| ECE | **0.024** | not available |
| Output tokens / decision | **0** | 3.48 |
| Time to first token (p50) | **0 ms** | ~100 ms |
| Latency p95 | **~1.1 s** | ~5.3 s |

Zero-training native-logits readout on the same suite: 88.3% accuracy, ECE 0.097.

## Training recipe

- Backbone: `Qwen/Qwen3-1.7B` (frozen, bfloat16), hidden size 2048.
- Head: shared scalar + set attention, ~0.8M parameters.
- Data: 8000 training requests converted from permissive public sources
  (banking77, CLINC150, Amazon Polarity, GSM8K verification; OOD evaluation uses
  MASSIVE and WANLI).
- Cost: ~37 minutes and ~4 GB peak memory on an Apple Silicon laptop; no GPU.
- Calibration: per-primitive temperature scaling fitted on dev outcomes.

## Usage

```python
from huggingface_hub import snapshot_download
from minojev import DecisionModel, Request, make_choice_question, ScoreOptions

path = snapshot_download("{repo_id}", allow_patterns=["general/*"])
model = DecisionModel.load(f"{{path}}/general", device="cpu")

request = Request(
    id="r1",
    state={{"text": "How do I add an existing card to the app?"}},
    questions=[make_choice_question("q", "Which category applies?",
                                    {{"card_linking": "card_linking", "atm": "atm", "transfer": "transfer"}})],
)
record = model.score([request], ScoreOptions(mode="reuse"))[0]
print(record["candidate_ids"], record["probabilities"], record["decode_steps"])
```

## Data and licensing

| Source | License | Use |
|---|---|---|
| mteb/banking77 (upstream PolyAI/banking77) | MIT (upstream CC-BY-4.0) | train |
| clinc/clinc_oos | CC-BY-3.0 | train |
| fancyzhx/amazon_polarity | Apache-2.0 | train |
| openai/gsm8k | MIT | train |
| SetFit/amazon_massive_intent_en-US (upstream MASSIVE) | CC-BY-4.0 | OOD evaluation |
| alisawuffles/WANLI | CC-BY-4.0 | OOD evaluation |

No NC-licensed data was used for training. The base model is Apache-2.0; this
checkpoint inherits that license. Code is MIT.

## Links

- Code, demos, and reproducible pipeline: https://github.com/{owner}/minojev
- Live benchmark page: https://{owner}.github.io/minojev/benchmark.html
- Datasets: https://huggingface.co/datasets/{owner}/minojev-data
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs/general-head-goldcal")
    parser.add_argument("--output", default="dist/hf-general")
    parser.add_argument("--repo-id", default="zeredy879/minojev")
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        shutil.rmtree(output)
    (output / "general").parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(Path(args.checkpoint), output / "general")
    card = MODEL_CARD.format(repo_id=args.repo_id, owner=args.repo_id.split("/")[0])
    (output / "README.md").write_text(card)
    metrics = {"general_test_200": {}}
    metrics_path = Path("results/general-metrics.json")
    if metrics_path.exists():
        data = json.loads(metrics_path.read_text())
        metrics["general_test_200"] = {
            "accuracy": data.get("accuracy"),
            "ece": data.get("expected_calibration_error"),
            "questions": data.get("questions"),
        }
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(output), "variant": "general", "repo_id": args.repo_id}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
