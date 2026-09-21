#!/usr/bin/env python3
"""Render an SVG preview of the first successful Sokoban run from a replay bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CELL = 40
DIRECTION_DELTAS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


def render(run: dict) -> str:
    size = run["size"]
    extent = size * CELL
    walls = {tuple(cell) for cell in run["walls"]}
    goals = {tuple(cell) for cell in run["goals"]}
    boxes = {tuple(cell) for cell in run["final"]["boxes"]}
    player = tuple(run["final"]["player"])
    font = "ui-monospace, Menlo, monospace"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {extent} {extent}" width="{extent}" height="{extent}">',
        f'<rect width="{extent}" height="{extent}" fill="#0a0d14"/>',
    ]
    for row in range(size):
        for column in range(size):
            cell = (row, column)
            if cell in walls:
                fill = "#2c3752"
            elif cell in goals:
                fill = "#16243a"
            else:
                fill = "#101728"
            parts.append(
                f'<rect x="{column * CELL + 2}" y="{row * CELL + 2}" width="{CELL - 4}" height="{CELL - 4}" rx="4" fill="{fill}"/>'
            )
            if cell in goals:
                parts.append(
                    f'<rect x="{column * CELL + 12}" y="{row * CELL + 12}" width="{CELL - 24}" height="{CELL - 24}" rx="3" fill="none" stroke="#f0b06a" stroke-width="2.4"/>'
                )
    path_cells = [tuple(run["start"]["player"])] + [tuple(frame["player"]) for frame in run["frames"]] + [player]
    if len(path_cells) > 1:
        trail = " ".join(f"{column * CELL + CELL / 2},{row * CELL + CELL / 2}" for row, column in path_cells)
        parts.append(
            f'<polyline points="{trail}" fill="none" stroke="#6ea8fe" stroke-opacity="0.4" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>'
        )
    for row, column in sorted(boxes):
        on_goal = (row, column) in goals
        stroke = "#59d3a8" if on_goal else "#f0b06a"
        fill = "#1f4d3d" if on_goal else "#7a5426"
        parts.append(
            f'<rect x="{column * CELL + 9}" y="{row * CELL + 9}" width="{CELL - 18}" height="{CELL - 18}" rx="4" fill="{fill}" stroke="{stroke}" stroke-width="2.4"/>'
        )
    parts.append(
        f'<circle cx="{player[1] * CELL + CELL / 2}" cy="{player[0] * CELL + CELL / 2}" r="{CELL * 0.22:.1f}" fill="#e8edf7" stroke="#0a0d14" stroke-width="3"/>'
    )
    optimal = run.get("optimal_steps")
    caption = (
        f'{run.get("id", "run")} · {"solved" if run["success"] else "unfinished"} in {run["steps"]} steps'
        + (f" (optimal {optimal})" if optimal is not None else "")
    )
    parts.append(
        f'<text x="10" y="{extent - 10}" fill="#8d9bb3" font-family="{font}" font-size="11">{caption}</text>'
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
