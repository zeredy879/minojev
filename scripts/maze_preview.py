#!/usr/bin/env python3
"""Render an SVG preview of the first successful maze run from a replay bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CELL = 40
DIRECTION_DELTAS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


def next_agent(frame: dict) -> tuple[int, int]:
    row, column = frame["agent"]
    delta_row, delta_column = DIRECTION_DELTAS[frame["move"]]
    return row + delta_row, column + delta_column


def render(run: dict) -> str:
    size = run["size"]
    extent = size * CELL
    walls = {(row, column) for row, column in run["walls"]}
    points = [tuple(run["start"])]
    for frame in run["frames"]:
        points.append(next_agent(frame))
    font = "ui-monospace, Menlo, monospace"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {extent} {extent}" width="{extent}" height="{extent}">',
        f'<rect width="{extent}" height="{extent}" fill="#0a0d14"/>',
    ]
    for row in range(size):
        for column in range(size):
            fill = "#2c3752" if (row, column) in walls else "#101728"
            parts.append(
                f'<rect x="{column * CELL + 2}" y="{row * CELL + 2}" width="{CELL - 4}" height="{CELL - 4}" rx="4" fill="{fill}"/>'
            )
    trail = " ".join(f"{column * CELL + CELL / 2},{row * CELL + CELL / 2}" for row, column in points)
    parts.append(f'<polyline points="{trail}" fill="none" stroke="#59d3a8" stroke-opacity="0.55" stroke-width="3" stroke-linejoin="round"/>')
    goal_row, goal_column = run["goal"]
    parts.append(
        f'<circle cx="{goal_column * CELL + CELL / 2}" cy="{goal_row * CELL + CELL / 2}" r="{CELL * 0.3:.1f}" fill="none" stroke="#f0b06a" stroke-width="2.4"/>'
    )
    start_row, start_column = run["start"]
    parts.append(
        f'<rect x="{start_column * CELL + CELL * 0.28:.1f}" y="{start_row * CELL + CELL * 0.28:.1f}" width="{CELL * 0.44:.1f}" height="{CELL * 0.44:.1f}" rx="3" fill="none" stroke="#6ea8fe" stroke-width="2.4"/>'
    )
    agent_row, agent_column = points[-1]
    parts.append(
        f'<circle cx="{agent_column * CELL + CELL / 2}" cy="{agent_row * CELL + CELL / 2}" r="{CELL * 0.22:.1f}" fill="#59d3a8" stroke="#0a0d14" stroke-width="3"/>'
    )
    parts.append(
        f'<text x="10" y="{extent - 10}" fill="#8d9bb3" font-family="{font}" font-size="11">'
        f'{run.get("id", "run")} · {run["steps"]} steps (optimal {run["optimal_steps"]})</text>'
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
