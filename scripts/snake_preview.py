#!/usr/bin/env python3
"""Render an SVG preview of the first successful snake run from a replay bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CELL = 40


def render(run: dict) -> str:
    size = run["size"]
    extent = size * CELL
    frames = run["frames"]
    body = [tuple(cell) for cell in run["final"]]
    heads = [tuple(run["start"][0])] + [tuple(frame["snake"][0]) for frame in frames] + [tuple(run["final"][0])]
    foods = []
    for frame in frames:
        cell = tuple(frame["food"])
        if cell not in foods:
            foods.append(cell)
    final_food = tuple(run["final_food"])
    if final_food not in foods:
        foods.append(final_food)
    font = "ui-monospace, Menlo, monospace"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {extent} {extent}" width="{extent}" height="{extent}">',
        f'<rect width="{extent}" height="{extent}" fill="#0a0d14"/>',
    ]
    for row in range(size):
        for column in range(size):
            parts.append(
                f'<rect x="{column * CELL + 2}" y="{row * CELL + 2}" width="{CELL - 4}" height="{CELL - 4}" rx="4" fill="#101728"/>'
            )
    for row, column in foods:
        parts.append(
            f'<circle cx="{column * CELL + CELL / 2}" cy="{row * CELL + CELL / 2}" r="{CELL * 0.16:.1f}" fill="#f0b06a" fill-opacity="0.85"/>'
        )
    if len(heads) > 1:
        trail = " ".join(f"{column * CELL + CELL / 2},{row * CELL + CELL / 2}" for row, column in heads)
        parts.append(
            f'<polyline points="{trail}" fill="none" stroke="#6ea8fe" stroke-opacity="0.45" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>'
        )
    if len(body) > 1:
        points = " ".join(f"{column * CELL + CELL / 2},{row * CELL + CELL / 2}" for row, column in body)
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="#59d3a8" stroke-opacity="0.85" stroke-width="10" stroke-linejoin="round" stroke-linecap="round"/>'
        )
    head_row, head_column = body[0]
    parts.append(
        f'<circle cx="{head_column * CELL + CELL / 2}" cy="{head_row * CELL + CELL / 2}" r="{CELL * 0.24:.1f}" fill="#b8f4dd" stroke="#0a0d14" stroke-width="3"/>'
    )
    parts.append(
        f'<text x="10" y="{extent - 10}" fill="#8d9bb3" font-family="{font}" font-size="11">'
        f'{run.get("id", "run")} · {run["eaten"]} food · {run["steps"]} steps</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    bundle = json.loads(Path(args.input).read_text())
    runs = bundle["runs"]
    run = next((candidate for candidate in runs if candidate["success"]), runs[0])
    Path(args.output).write_text(render(run) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
