#!/usr/bin/env python3
"""Assemble web/data/calibration.json from committed calibrated predictions."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "general-predictions.jsonl"
OUTPUT = ROOT / "web" / "data" / "calibration.json"


def main() -> int:
    records = []
    for line in Path(SOURCE).read_text(encoding="utf-8").split("\n"):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if "correct" not in row or not row.get("probabilities"):
            continue
        confidence = max(row["probabilities"])
        records.append(
            {
                "id": row["id"],
                "qid": row.get("qid"),
                "type": row.get("type"),
                "family": row.get("family"),
                "confidence": round(confidence, 6),
                "correct": bool(row["correct"]),
            }
        )
    total = len(records)
    accuracy = sum(record["correct"] for record in records) / max(total, 1)
    bins = []
    for index in range(10):
        low, high = index / 10, (index + 1) / 10
        members = [
            record
            for record in records
            if (low < record["confidence"] <= high) or (index == 0 and record["confidence"] <= high)
        ]
        if members:
            bins.append(
                {
                    "low": low,
                    "high": high,
                    "count": len(members),
                    "mean_confidence": round(sum(record["confidence"] for record in members) / len(members), 4),
                    "accuracy": round(sum(record["correct"] for record in members) / len(members), 4),
                }
            )
    ece = sum(bin_entry["count"] / total * abs(bin_entry["mean_confidence"] - bin_entry["accuracy"]) for bin_entry in bins)
    thresholds = {}
    for threshold in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        members = [record for record in records if record["confidence"] >= threshold]
        if members:
            thresholds[str(threshold)] = {
                "coverage": round(len(members) / total, 4),
                "accuracy": round(sum(record["correct"] for record in members) / len(members), 4),
            }
    bundle = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": "Qwen3-1.7B head-trained, gold-calibrated",
        "suite": "200 balanced requests (banking77, CLINC150, Amazon Polarity, GSM8K verification)",
        "questions": total,
        "accuracy": round(accuracy, 4),
        "ece": round(ece, 4),
        "bins": bins,
        "thresholds": thresholds,
        "points": records,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(OUTPUT), "questions": total, "accuracy": bundle["accuracy"], "ece": bundle["ece"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
