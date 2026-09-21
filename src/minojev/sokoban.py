"""Sokoban decision task, solver, and agent rollout.

A Sokoban state is structured JSON: grid size, player, boxes, goals, and the
wall list. Each step asks six independent questions in one forward pass:

- one choice over the four move directions (teacher: moves that keep the
  puzzle solvable are preferred; deadlocks score near zero);
- four booleans, one per direction ("is moving up safe?"), where unsafe means
  the move is illegal or creates a corner-deadlocked box;
- one score over progress levels ("how many boxes are on targets?").

Levels are generated backwards from a solved state, so every generated level
is guaranteed solvable. A breadth-first solver over (boxes, player) states
provides both the optimal first move for supervision and the optimal step
count shown in the replay.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, replace

from .model import ScoreOptions
from .types import Request, make_boolean_question, make_choice_question, make_score_question

DIRECTIONS = ("up", "down", "left", "right")
DELTAS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

SOLVER_MAX_STATES = 200_000


@dataclass(frozen=True)
class Sokoban:
    size: int
    walls: frozenset[tuple[int, int]]
    goals: frozenset[tuple[int, int]]
    boxes: frozenset[tuple[int, int]]
    player: tuple[int, int]


def in_bounds(size: int, cell: tuple[int, int]) -> bool:
    row, column = cell
    return 0 <= row < size and 0 <= column < size


def step_cell(cell: tuple[int, int], direction: str) -> tuple[int, int]:
    delta = DELTAS[direction]
    return cell[0] + delta[0], cell[1] + delta[1]


def blocked_cell(level: Sokoban, cell: tuple[int, int]) -> bool:
    return not in_bounds(level.size, cell) or cell in level.walls


def free_cell(level: Sokoban, cell: tuple[int, int]) -> bool:
    return not blocked_cell(level, cell) and cell not in level.boxes


def on_target(level: Sokoban) -> int:
    return len(level.boxes & level.goals)


def solved(level: Sokoban) -> bool:
    return level.boxes <= level.goals


def push_target(level: Sokoban, direction: str) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Return (player target, box target) when the move is a legal push."""
    target = step_cell(level.player, direction)
    if target not in level.boxes:
        return None
    beyond = step_cell(target, direction)
    if not free_cell(level, beyond):
        return None
    return target, beyond


def move_legal(level: Sokoban, direction: str) -> bool:
    target = step_cell(level.player, direction)
    if blocked_cell(level, target):
        return False
    if target in level.boxes:
        return push_target(level, direction) is not None
    return True


def apply_move(level: Sokoban, direction: str) -> Sokoban:
    target = step_cell(level.player, direction)
    if target in level.boxes:
        beyond = step_cell(target, direction)
        boxes = frozenset((level.boxes - {target}) | {beyond})
        return replace(level, boxes=boxes, player=target)
    return replace(level, player=target)


def corner_deadlock(level: Sokoban, box: tuple[int, int]) -> bool:
    if box in level.goals:
        return False
    vertical = blocked_cell(level, (box[0] - 1, box[1])) or blocked_cell(level, (box[0] + 1, box[1]))
    horizontal = blocked_cell(level, (box[0], box[1] - 1)) or blocked_cell(level, (box[0], box[1] + 1))
    return vertical and horizontal


def deadlocked(level: Sokoban) -> bool:
    return any(corner_deadlock(level, box) for box in level.boxes)


def safe_move(level: Sokoban, direction: str) -> bool:
    if not move_legal(level, direction):
        return False
    return not deadlocked(apply_move(level, direction))


def _cell_index(level: Sokoban, cell: tuple[int, int]) -> int:
    return cell[0] * level.size + cell[1]


def _index_cell(level: Sokoban, index: int) -> tuple[int, int]:
    return divmod(index, level.size)


def _cell_mask(level: Sokoban, cells) -> int:
    mask = 0
    for cell in cells:
        mask |= 1 << _cell_index(level, cell)
    return mask


def solve(level: Sokoban, max_depth: int | None = None, max_states: int = SOLVER_MAX_STATES) -> list[str] | None:
    """Shortest walk-and-push sequence that puts every box on a goal.

    ``max_depth`` bounds the search; ``None`` searches until the state graph is
    exhausted. Returns ``None`` when no solution exists within the bounds.
    """
    goals_mask = _cell_mask(level, level.goals)
    walls_mask = _cell_mask(level, level.walls)
    start_boxes = _cell_mask(level, level.boxes)
    if start_boxes & ~goals_mask == 0:
        return []
    start = (start_boxes, _cell_index(level, level.player))
    parents: dict[tuple[int, int], tuple[tuple[int, int], str] | None] = {start: None}
    depths = {start: 0}
    queue = deque([start])
    size = level.size

    def free(index: int, boxes: int) -> bool:
        if index < 0 or index >= size * size:
            return False
        bit = 1 << index
        return not (walls_mask & bit) and not (boxes & bit)

    while queue:
        boxes, player = queue.popleft()
        depth = depths[(boxes, player)]
        if max_depth is not None and depth >= max_depth:
            continue
        row, column = divmod(player, size)
        for direction, (delta_row, delta_column) in DELTAS.items():
            next_row, next_column = row + delta_row, column + delta_column
            if not (0 <= next_row < size and 0 <= next_column < size):
                continue
            target = next_row * size + next_column
            if walls_mask & (1 << target):
                continue
            next_boxes = boxes
            bit = 1 << target
            if boxes & bit:
                beyond_row, beyond_column = next_row + delta_row, next_column + delta_column
                if not (0 <= beyond_row < size and 0 <= beyond_column < size):
                    continue
                beyond = beyond_row * size + beyond_column
                if not free(beyond, boxes):
                    continue
                next_boxes = (boxes & ~bit) | (1 << beyond)
            state = (next_boxes, target)
            if state in parents:
                continue
            parents[state] = ((boxes, player), direction)
            depths[state] = depth + 1
            if next_boxes & ~goals_mask == 0:
                return _reconstruct(parents, state)
            if len(parents) > max_states:
                return None
            queue.append(state)
    return None


def _reconstruct(parents: dict, state) -> list[str]:
    moves: list[str] = []
    while parents[state] is not None:
        previous, direction = parents[state]
        moves.append(direction)
        state = previous
    moves.reverse()
    return moves


def _reverse_scramble(level: Sokoban, rng: random.Random, steps: int) -> Sokoban:
    """Reverse-play the level: walk the player and pull boxes off their goals.

    Every reverse move inverts a legal forward move, so the result is always
    solvable by replaying the scramble backwards.
    """
    player = level.player
    boxes = set(level.boxes)
    for _ in range(steps):
        options = []
        for direction, (delta_row, delta_column) in DELTAS.items():
            target = (player[0] + delta_row, player[1] + delta_column)
            if target in boxes:
                away = (player[0] - delta_row, player[1] - delta_column)
                if not blocked_cell(level, away) and away not in boxes:
                    options.append((direction, target, away))
            elif not blocked_cell(level, target):
                options.append((direction, target, None))
        if not options:
            break
        _, target, away = rng.choice(options)
        if away is None:
            player = target
            continue
        boxes.remove(target)
        boxes.add(player)
        player = away
    return replace(level, boxes=frozenset(boxes), player=player)


def generate_ready_level(rng: random.Random, size: int = 7) -> Sokoban:
    """A one-push puzzle: the box is one push from its goal and the player is
    already positioned behind it. The model must still pick the push side."""
    interior = range(1, size - 1)
    while True:
        box = (rng.choice(interior), rng.choice(interior))
        direction = rng.choice(DIRECTIONS)
        delta_row, delta_column = DELTAS[direction]
        goal = (box[0] + delta_row, box[1] + delta_column)
        player = (box[0] - delta_row, box[1] - delta_column)
        if all(1 <= coordinate < size - 1 for coordinate in (goal[0], goal[1], player[0], player[1])):
            walls = frozenset(
                (row, column)
                for row in range(size)
                for column in range(size)
                if row in (0, size - 1) or column in (0, size - 1)
            )
            return Sokoban(
                size=size,
                walls=walls,
                goals=frozenset({goal}),
                boxes=frozenset({box}),
                player=player,
            )


def generate_sokoban(
    rng: random.Random,
    size: int = 7,
    boxes: int = 1,
    scramble_steps: int = 15,
    min_solution: int = 2,
    wall_density: float = 0.08,
    attempts: int = 300,
    ready: bool = False,
) -> Sokoban:
    if ready and boxes == 1:
        return generate_ready_level(rng, size=size)
    inner = [(row, column) for row in range(1, size - 1) for column in range(1, size - 1)]
    for _ in range(attempts):
        walls = {
            (row, column)
            for row in range(size)
            for column in range(size)
            if row in (0, size - 1) or column in (0, size - 1)
        }
        for cell in inner:
            if rng.random() < wall_density:
                walls.add(cell)
        free = [cell for cell in inner if cell not in walls]
        if len(free) < boxes + 5:
            continue
        goals = frozenset(rng.sample(free, boxes))
        anchored = frozenset(goals)
        candidates = [
            cell
            for cell in free
            if cell not in anchored
            and any(step_cell(cell, direction) in anchored for direction in DIRECTIONS)
        ]
        if not candidates:
            continue
        level = Sokoban(
            size=size,
            walls=frozenset(walls),
            goals=goals,
            boxes=goals,
            player=rng.choice(candidates),
        )
        level = _reverse_scramble(level, rng, scramble_steps)
        if solved(level):
            continue
        solution = solve(level)
        if solution is None or len(solution) < min_solution:
            continue
        return level
    raise RuntimeError("could not generate a solvable Sokoban level")


def _walk_state(rng: random.Random, level: Sokoban, steps: int) -> Sokoban:
    for _ in range(steps):
        options = [direction for direction in DIRECTIONS if safe_move(level, direction)]
        if not options:
            break
        level = apply_move(level, rng.choice(options))
    return level


def _move_teacher(level: Sokoban, solution: list[str], temperature: float = 2.0) -> dict[str, float]:
    """Dense solver supervision: moves that leave shorter solutions score higher."""
    distances = {}
    for direction in DIRECTIONS:
        if not move_legal(level, direction):
            continue
        moved = apply_move(level, direction)
        if deadlocked(moved):
            continue
        rest = solve(moved)
        if rest is None:
            continue
        distances[direction] = len(rest)
    weights = {direction: 1e-6 for direction in DIRECTIONS}
    if distances:
        best = min(distances.values())
        for direction, value in distances.items():
            weights[direction] = math.exp(-(value - best) / temperature)
    total = sum(weights.values())
    return {direction: weights[direction] / total for direction in DIRECTIONS}


def _score_levels(boxes: int) -> list[str]:
    box_word = "box" if boxes == 1 else "boxes"
    target_word = "target" if boxes == 1 else "targets"
    return [f"{index} of {boxes} {box_word} on {target_word}" for index in range(boxes + 1)]


def state_object(level: Sokoban, step: int = 0) -> str:
    """Compact Sokoban map plus explicit coordinates.

    Map legend: ``#`` wall, ``.`` floor, ``o`` target, ``$`` box, ``*`` box on
    target, ``@`` player, ``+`` player on target. The coordinate lines matter:
    a tiny model reads exact positions far more reliably than a raw grid.
    """
    rows = []
    for row in range(level.size):
        cells = []
        for column in range(level.size):
            cell = (row, column)
            if cell in level.walls:
                cells.append("#")
            elif cell in level.boxes:
                cells.append("*" if cell in level.goals else "$")
            elif cell == level.player:
                cells.append("+" if cell in level.goals else "@")
            elif cell in level.goals:
                cells.append("o")
            else:
                cells.append(".")
        rows.append("".join(cells))
    boxes = " ".join(f"{row},{column}" for row, column in sorted(level.boxes))
    goals = " ".join(f"{row},{column}" for row, column in sorted(level.goals))
    return (
        f"{level.size}x{level.size}\n"
        + "\n".join(rows)
        + f"\nplayer {level.player[0]},{level.player[1]}\nboxes {boxes}\ngoals {goals}"
    )


def state_request(level: Sokoban, step: int = 0, solution: list[str] | None = None, with_teacher: bool = True) -> Request:
    if solution is None:
        solution = solve(level)
    if not solution:
        raise ValueError("state_request needs a solvable, unfinished Sokoban state")
    questions = [
        make_choice_question(
            "move",
            "Which move should the player make to push every box onto a target?",
            {direction: f"move {direction}" for direction in DIRECTIONS},
        ),
    ]
    gold: dict = {"move": solution[0]}
    teacher: dict = {"move": _move_teacher(level, solution)} if with_teacher else {}
    for direction in DIRECTIONS:
        qid = f"safe-{direction}"
        safe = safe_move(level, direction)
        questions.append(
            make_boolean_question(
                qid,
                f"Is moving {direction} safe?",
                {
                    "true": f"moving {direction} is legal and does not deadlock a box",
                    "false": f"moving {direction} is blocked or deadlocks a box",
                },
            )
        )
        gold[qid] = safe
        if with_teacher:
            teacher[qid] = {"true": 0.9 if safe else 0.1, "false": 0.1 if safe else 0.9}
    boxes = len(level.boxes)
    questions.append(
        make_score_question("progress", "How many boxes are on targets?", _score_levels(boxes))
    )
    current = on_target(level)
    gold["progress"] = current
    if with_teacher:
        level_target = [0.04] * (boxes + 1)
        level_target[current] = 0.84
        teacher["progress"] = {str(index): value for index, value in enumerate(level_target)}
    for question in questions:
        question.family = question.kind
    return Request(
        id=f"sokoban-{level.size}-{level.player[0]}-{level.player[1]}-{step}",
        state=state_object(level, step),
        questions=questions,
        gold=gold,
        teacher=teacher,
    )


def generate_request(
    index: int,
    seed: int = 17,
    split: str = "train",
    size: int = 7,
    boxes: int = 1,
    ready: bool = True,
) -> Request:
    offsets = {"train": 0, "dev": 1_000_000, "test": 2_000_000}
    rng = random.Random((seed * 5_237) ^ (index + offsets.get(split, 3_000_000)))
    for _ in range(60):
        try:
            level = generate_sokoban(rng, size=size, boxes=boxes, ready=ready)
        except RuntimeError:
            continue
        if ready:
            state = level if rng.random() < 0.6 else _walk_state(rng, level, rng.randint(1, 4))
            if solved(state):
                continue
            rest = solve(state)
            if not rest:
                continue
            return state_request(state, step=rng.randint(0, 40), solution=rest)
        solution = solve(level)
        if not solution:
            continue
        if rng.random() < 0.7:
            prefix = rng.randint(1, max(1, len(solution) - 1))
            state = level
            for direction in solution[:prefix]:
                state = apply_move(state, direction)
        else:
            state = _walk_state(rng, level, rng.randint(0, 18))
        if solved(state):
            continue
        rest = solve(state)
        if not rest:
            continue
        return state_request(state, step=rng.randint(0, 40), solution=rest)
    raise RuntimeError("could not generate a Sokoban request")


def generate_split(
    count: int,
    seed: int = 17,
    split: str = "train",
    size: int = 7,
    boxes: int = 1,
    ready: bool = True,
) -> list[Request]:
    return [generate_request(index, seed=seed, split=split, size=size, boxes=boxes, ready=ready) for index in range(count)]


def rollout(
    model,
    size: int = 7,
    seed: int = 29,
    boxes: int = 1,
    max_steps: int = 90,
    device: str = "auto",
    ready: bool = True,
) -> dict:
    rng = random.Random(seed)
    level = generate_sokoban(rng, size=size, boxes=boxes, ready=ready)
    initial = level
    optimal = solve(initial)
    frames = []
    success = False
    for step in range(max_steps):
        solution = solve(level)
        if solution is None:
            break
        request = state_request(level, step, solution=solution, with_teacher=False)
        records = {record["qid"]: record for record in model.score([request], ScoreOptions(device=device))}
        choice = records["move"]
        combined = {}
        for direction in DIRECTIONS:
            safety = records[f"safe-{direction}"]["probabilities"][1]
            combined[direction] = choice["probabilities"][choice["candidate_ids"].index(direction)] * safety
        safe = [direction for direction in DIRECTIONS if safe_move(level, direction)]
        if not safe:
            break
        best_legal = max(safe, key=lambda direction: combined[direction])
        greedy_choice = max(
            DIRECTIONS,
            key=lambda direction: choice["probabilities"][choice["candidate_ids"].index(direction)],
        )
        forced = greedy_choice not in safe
        move = best_legal
        before = level
        level = apply_move(before, move)
        frames.append(_frame(step, before, move, forced, choice, records, combined))
        if solved(level):
            success = True
            break
    return {
        "size": size,
        "walls": [[row, column] for row, column in sorted(initial.walls)],
        "goals": [[row, column] for row, column in sorted(initial.goals)],
        "start": {
            "player": [initial.player[0], initial.player[1]],
            "boxes": [[row, column] for row, column in sorted(initial.boxes)],
        },
        "success": success,
        "steps": len(frames),
        "optimal_steps": len(optimal) if optimal is not None else None,
        "final": {
            "player": [level.player[0], level.player[1]],
            "boxes": [[row, column] for row, column in sorted(level.boxes)],
        },
        "frames": frames,
    }


def _frame(step: int, level: Sokoban, move: str, forced: bool, choice: dict, records: dict, combined: dict) -> dict:
    return {
        "step": step,
        "player": [level.player[0], level.player[1]],
        "boxes": [[row, column] for row, column in sorted(level.boxes)],
        "move": move,
        "forced": forced,
        "on_target": on_target(level),
        "pushing": step_cell(level.player, move) in level.boxes,
        "move_probabilities": {
            direction: round(choice["probabilities"][choice["candidate_ids"].index(direction)], 6)
            for direction in DIRECTIONS
        },
        "safety": {direction: round(records[f"safe-{direction}"]["probabilities"][1], 6) for direction in DIRECTIONS},
        "combined": {direction: round(combined[direction], 6) for direction in DIRECTIONS},
        "progress_expected": records["progress"]["expected"],
    }


def rollout_bundle(
    model,
    count: int = 8,
    seed: int = 11,
    size: int = 7,
    boxes: int = 1,
    max_steps: int = 90,
    device: str = "auto",
    ready: bool = True,
) -> dict:
    runs = []
    for index in range(count):
        result = rollout(
            model,
            size=size,
            seed=seed * 1_000 + index,
            boxes=boxes,
            max_steps=max_steps,
            device=device,
            ready=ready,
        )
        result["id"] = f"sokoban-{index + 1}"
        runs.append(result)
    return {
        "runs": runs,
        "summary": {
            "levels": len(runs),
            "successes": sum(run["success"] for run in runs),
            "total_steps": sum(run["steps"] for run in runs),
            "forced_moves": sum(frame["forced"] for run in runs for frame in run["frames"]),
        },
    }
