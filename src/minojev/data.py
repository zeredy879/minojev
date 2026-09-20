"""JSONL readers and writers for requests and scored records."""

from __future__ import annotations

import json
from pathlib import Path

from .types import Request, request_from_object, request_to_object


def read_requests(path: str | Path) -> list[Request]:
    # Split on "\n" only: str.splitlines() also breaks on Unicode line
    # separators such as U+2028, which can appear inside JSON strings.
    requests = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").split("\n")):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        requests.append(request_from_object(json.loads(line), index=number))
    if not requests:
        raise ValueError(f"No requests found in {path}")
    return requests


def read_records(path: str | Path) -> list[dict]:
    records = []
    for line in Path(path).read_text(encoding="utf-8").split("\n"):
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def write_jsonl(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_requests(requests: list[Request], path: str | Path) -> None:
    write_jsonl([request_to_object(request) for request in requests], path)
