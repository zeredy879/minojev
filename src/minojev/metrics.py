"""Aggregate metrics for scored decision records."""

from __future__ import annotations

import math

from .types import BOOLEAN


def _teacher_vectors(record: dict) -> tuple[list[float], list[float]] | None:
    teacher = record.get("teacher")
    if not teacher:
        return None
    ids = record["candidate_ids"]
    if set(teacher) != set(ids):
        return None
    target = [float(teacher[candidate_id]) for candidate_id in ids]
    total = sum(target)
    if total <= 0:
        return None
    return [value / total for value in target], record["probabilities"]


def aggregate(records: list[dict]) -> dict:
    total = len(records)
    summary: dict = {"questions": total}
    labelled = [record for record in records if "correct" in record]
    if labelled:
        summary["labelled"] = len(labelled)
        summary["accuracy"] = sum(bool(record["correct"]) for record in labelled) / len(labelled)
        by_type: dict[str, list[bool]] = {}
        for record in labelled:
            by_type.setdefault(record["type"], []).append(bool(record["correct"]))
        summary["accuracy_by_type"] = {
            kind: round(sum(values) / len(values), 6) for kind, values in sorted(by_type.items())
        }
        by_family: dict[str, list[bool]] = {}
        for record in labelled:
            family = record.get("family")
            if family:
                by_family.setdefault(family, []).append(bool(record["correct"]))
        if by_family:
            summary["accuracy_by_family"] = {
                family: round(sum(values) / len(values), 6) for family, values in sorted(by_family.items())
            }
        nll_terms = []
        for record in labelled:
            gold = record["gold"]
            probabilities = record["probabilities"]
            if record["type"] == BOOLEAN:
                index = 1 if gold is True or gold in (1, "true", "True") else 0
            elif record["type"] == "score":
                index = int(gold)
            else:
                index = record["candidate_ids"].index(gold)
            nll_terms.append(-math.log(max(probabilities[index], 1e-12)))
        summary["gold_nll"] = sum(nll_terms) / len(nll_terms)
    pairs = []
    for record in records:
        vectors = _teacher_vectors(record)
        if vectors is not None:
            pairs.append((record, vectors))
    if pairs:
        summary["teacher_scored"] = len(pairs)
        summary["teacher_ce"] = sum(
            -sum(target * math.log(max(probability, 1e-12)) for target, probability in zip(targets, probabilities))
            for _, (targets, probabilities) in pairs
        ) / len(pairs)
        summary["distribution_error"] = sum(
            sum((probability - target) ** 2 for target, probability in zip(targets, probabilities))
            for _, (targets, probabilities) in pairs
        ) / len(pairs)
        summary["brier"] = summary["distribution_error"]
        agreement = []
        by_family: dict[str, list[bool]] = {}
        for record, (targets, probabilities) in pairs:
            matched = max(range(len(probabilities)), key=probabilities.__getitem__) == max(
                range(len(targets)), key=targets.__getitem__
            )
            agreement.append(matched)
            family = record.get("family")
            if family:
                by_family.setdefault(family, []).append(matched)
        summary["teacher_argmax_accuracy"] = sum(agreement) / len(agreement)
        topset = []
        by_family_topset: dict[str, list[bool]] = {}
        for record, (targets, probabilities) in pairs:
            best = max(targets)
            tied = {index for index, value in enumerate(targets) if abs(value - best) <= 1e-9}
            predicted = max(range(len(probabilities)), key=probabilities.__getitem__)
            matched = predicted in tied
            topset.append(matched)
            family = record.get("family")
            if family:
                by_family_topset.setdefault(family, []).append(matched)
        summary["teacher_topset_accuracy"] = sum(topset) / len(topset)
        if by_family:
            summary["teacher_argmax_by_family"] = {
                family: round(sum(values) / len(values), 6) for family, values in sorted(by_family.items())
            }
        if by_family_topset:
            summary["teacher_topset_by_family"] = {
                family: round(sum(values) / len(values), 6) for family, values in sorted(by_family_topset.items())
            }
    summary["decode_steps"] = sum(int(record.get("decode_steps", 0)) for record in records)
    return summary
