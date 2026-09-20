"""Build the general decision dataset from permissive public sources.

Splits are isolated by source: training sources contribute train/dev/test
rows, while out-of-domain sources are reserved for evaluation only. Every
written line keeps ``source`` and ``license`` for auditing.
"""

from __future__ import annotations

import itertools
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from .converters import (
    SourceSpec,
    classification_to_choice,
    continuous_to_boolean,
    continuous_to_score,
    convert_rows,
    nli_to_choice,
)
from .data import write_jsonl
from .types import Request, request_to_object

# Sources whose labels come from the upstream ClassLabel feature are marked
# labels=(); the loader fills them in from the dataset features.
SOURCES: dict[str, SourceSpec] = {
    "banking77": SourceSpec(
        name="mteb/banking77",
        license="mit",
        primitive="choice",
        group="train",
        text_field="text",
        label_field="label",
        label_text_field="label_text",
        notes="77 banking intents (mteb parquet mirror; upstream PolyAI/banking77)",
    ),
    "clinc_oos": SourceSpec(
        name="clinc/clinc_oos",
        license="cc-by-3.0",
        primitive="choice",
        group="train",
        config="plus",
        text_field="text",
        label_field="intent",
        notes="150 intents including out-of-scope",
    ),
    "amazon_polarity": SourceSpec(
        name="fancyzhx/amazon_polarity",
        license="apache-2.0",
        primitive="boolean",
        group="train",
        text_field="content",
        label_field="label",
        notes="binary sentiment as yes/no",
    ),
    "gsm8k": SourceSpec(
        name="openai/gsm8k",
        license="mit",
        primitive="verify",
        group="train",
        config="main",
        text_field="question",
        notes="is this solution correct? (wrong answers injected)",
    ),
    "massive": SourceSpec(
        name="SetFit/amazon_massive_intent_en-US",
        license="cc-by-4.0",
        primitive="choice",
        group="ood",
        text_field="text",
        label_field="label",
        label_text_field="label_text",
        notes="held out: 60 intents, spoken assistant (SetFit mirror of AmazonScience/massive)",
    ),
    "wanli": SourceSpec(
        name="alisawuffles/WANLI",
        license="cc-by-4.0",
        primitive="nli-choice",
        group="ood",
        text_field="premise",
        label_field="gold",
        labels=("entailment", "neutral", "contradiction"),
        notes="held out: adversarial NLI",
    ),
}


def _normalize_row(row: dict, spec: SourceSpec) -> dict:
    if spec.label_field in row and hasattr(row[spec.label_field], "item"):
        row[spec.label_field] = row[spec.label_field].item()
    return row


def _label_names(features, spec: SourceSpec) -> tuple[str, ...]:
    if spec.labels:
        return spec.labels
    feature = features.get(spec.label_field) if features else None
    names = getattr(feature, "names", None)
    if names:
        return tuple(str(name) for name in names)
    if spec.label_text_field:
        from datasets import load_dataset

        dataset = load_dataset(spec.name, spec.config, split=spec.split, streaming=True)
        values = set()
        for index, row in enumerate(dataset):
            if index >= 50000:
                break
            values.add(str(row[spec.label_text_field]))
        if not values:
            raise ValueError(f"Source {spec.name} produced no labels while scanning")
        return tuple(sorted(values))
    raise ValueError(f"Source {spec.name} has no ClassLabel names for {spec.label_field!r}")


def load_rows(spec: SourceSpec, limit: int):
    """Load up to ``limit`` rows via streaming and return rows plus features."""
    from datasets import load_dataset

    dataset = load_dataset(spec.name, spec.config, split=spec.split, streaming=True)
    rows = list(itertools.islice(dataset, limit))
    rows = [_normalize_row(dict(row), spec) for row in rows]
    return rows, dataset.features


def _gsm8k_verify(rows: list[dict], spec: SourceSpec, seed: int) -> list[Request]:
    """Correct versus perturbed final answers, as a boolean verification task."""
    rng = random.Random(seed)
    requests = []
    for index, row in enumerate(rows):
        answer = str(row.get("answer", ""))
        # GSM8K answers end with "#### <number>"; keep the reasoning, swap the result.
        marker = answer.rfind("####")
        correct = answer[marker + 4 :].strip() if marker >= 0 else answer.strip()
        solution = answer[:marker].strip() if marker >= 0 else answer
        digits = "".join(character for character in correct if character.isdigit())
        if digits and rng.random() < 0.5:
            wrong = str(int(digits) + rng.choice([1, 2, 3, 10]))
            claimed = wrong
            truth = False
        else:
            claimed = correct
            truth = True
        question = make_verify_question(claimed)
        question.family = spec.name
        requests.append(
            Request(
                id=f"{spec.name}-{index}",
                state={"problem": row[spec.text_field], "solution": solution, "claimed_answer": claimed},
                questions=[question],
                gold={"correct": truth},
                teacher={"correct": {"true": 0.9 if truth else 0.1, "false": 0.1 if truth else 0.9}},
            )
        )
    return requests


def make_verify_question(claimed: str):
    from .types import make_boolean_question

    return make_boolean_question(
        "correct",
        f"Is the claimed final answer '{claimed}' correct for this problem?",
        {"true": f"the claimed answer {claimed} is correct", "false": f"the claimed answer {claimed} is incorrect"},
    )


CUSTOM_CONVERTERS = {
    "verify": _gsm8k_verify,
}


def convert_source(spec: SourceSpec, rows: list[dict], features, seed: int) -> list[Request]:
    if spec.primitive in CUSTOM_CONVERTERS:
        return CUSTOM_CONVERTERS[spec.primitive](rows, spec, seed)
    if spec.primitive == "nli-choice":
        from .converters import nli_to_choice as converter

        labels = _label_names(features, spec)
        resolved = SourceSpec(**{**spec.__dict__, "labels": labels})
        return [converter(row, resolved, random.Random(seed + index)) for index, row in enumerate(rows)]
    if spec.primitive in ("continuous-score", "continuous-gate"):
        raise ValueError(f"{spec.name}: continuous sources need a custom converter")
    labels = _label_names(features, spec)
    resolved = SourceSpec(**{**spec.__dict__, "labels": labels})
    if spec.primitive == "choice":
        from .converters import classification_to_choice as converter
    elif spec.primitive == "boolean":
        from .converters import binary_to_boolean as converter
    else:
        raise ValueError(f"Unknown primitive {spec.primitive!r} for {spec.name}")
    rng = random.Random(seed)
    return [converter(row, resolved, rng) for row in rows]


@dataclass
class BuildConfig:
    out_dir: str = "data"
    per_source: int = 4000
    per_ood: int = 1000
    seed: int = 17
    sources: list[str] | None = None


def build_dataset(config: BuildConfig) -> dict:
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits: dict[str, list[tuple[Request, SourceSpec]]] = {"train": [], "dev": [], "test": [], "ood": []}
    card: dict = {"sources": [], "counts": {}}
    selected = config.sources or list(SOURCES)
    rng = random.Random(config.seed)
    failures = {}
    for key in selected:
        spec = SOURCES[key]
        limit = config.per_source if spec.group == "train" else config.per_ood
        try:
            rows, features = load_rows(spec, limit)
            requests = convert_source(spec, rows, features, config.seed + abs(hash(key)) % 1000)
        except Exception as error:  # network or schema issue: keep other sources
            failures[key] = f"{type(error).__name__}: {error}"
            continue
        if spec.group == "ood":
            splits["ood"].extend((request, spec) for request in requests)
        else:
            shuffled = requests[:]
            rng.shuffle(shuffled)
            train_end = int(len(shuffled) * 0.8)
            dev_end = int(len(shuffled) * 0.9)
            splits["train"].extend((request, spec) for request in shuffled[:train_end])
            splits["dev"].extend((request, spec) for request in shuffled[train_end:dev_end])
            splits["test"].extend((request, spec) for request in shuffled[dev_end:])
        card["sources"].append(
            {
                "key": key,
                "name": spec.name,
                "license": spec.license,
                "primitive": spec.primitive,
                "group": spec.group,
                "rows": len(requests),
                "notes": spec.notes,
            }
        )
    written = {}
    for split_name, entries in splits.items():
        rows = []
        for request, spec in entries:
            row = request_to_object(request)
            row["source"] = spec.name
            row["license"] = spec.license
            rows.append(row)
        path = out_dir / f"general-{split_name}.jsonl"
        write_jsonl(rows, path)
        written[split_name] = len(rows)
    card["counts"] = written
    card["failures"] = failures
    (out_dir / "general-dataset-card.json").write_text(json.dumps(card, ensure_ascii=False, indent=2) + "\n")
    return {
        "splits": written,
        "sources": len(card["sources"]),
        "failures": failures,
        "card": str(out_dir / "general-dataset-card.json"),
    }
